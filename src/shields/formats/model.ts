import { SeededPrng } from "../../core/prng.js";
import type { ProtectionIntensity } from "../../core/types.js";
import { generateChaffPayloads } from "../chaff/chaff-generator.js";

export interface ModelInspectionSummary {
  readonly validModel: boolean;
  readonly format: "safetensors" | "gguf" | "unknown";
  readonly tensorCount: number;
  readonly metadataCount: number;
  readonly exceedsReaNestingLimit: boolean;
  readonly blaProtected: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Invalid or unsupported input is never replaced with synthetic weights. */
export class InvalidSafetensorsError extends Error {
  constructor(reason: string) {
    super(`Invalid or unsupported Safetensors input: ${reason}`);
    this.name = "InvalidSafetensorsError";
  }
}

// Conservative supported dtype subset. Unknown/new dtypes fail closed.
const DTYPE_BYTES: Readonly<Record<string, number>> = Object.freeze({
  BOOL: 1, U8: 1, I8: 1, I16: 2, U16: 2, F16: 2, BF16: 2,
  I32: 4, U32: 4, F32: 4, I64: 8, U64: 8, F64: 8,
});

/** Structural validation only: this does not run model inference. */
export function validateSafetensorsBuffer(bytes: Buffer): {
  header: Record<string, unknown>; tensorData: Buffer;
} {
  const fail = (reason: string): never => { throw new InvalidSafetensorsError(reason); };
  if (bytes.length < 10) fail("truncated header");
  const length = bytes.readBigUInt64LE(0);
  if (length > 100_000_000n || length > BigInt(bytes.length - 8) || length < 2n) {
    fail("invalid header length");
  }
  const end = 8 + Number(length);
  let text: string;
  let header: unknown;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes.subarray(8, end));
    if (!text.startsWith("{")) fail("header must start with an object");
    header = JSON.parse(text);
    // JSON.parse otherwise silently loses duplicate fields, including tensor names.
    const stack: Array<Set<string> | null> = [];
    const tokens = /"(?:[^"\\]|\\.)*"|[{}\[\]]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
    for (const token of text.matchAll(tokens)) {
      const value = token[0];
      if (value === "{") stack.push(new Set());
      else if (value === "[") stack.push(null);
      else if (value === "}" || value === "]") stack.pop();
      else if (!value.startsWith('"')) {
        // JSON.parse rounds before Number.isSafeInteger can inspect the value.
        // The supported header subset requires integer lexical tokens, not
        // fractions/exponents even when they would round to an integer.
        if (!/^\d+$/.test(value) || BigInt(value) > BigInt(Number.MAX_SAFE_INTEGER)) {
          fail("unsupported numeric token: require nonnegative safe integer notation");
        }
      } else {
        const decoded = JSON.parse(value) as string;
        if (/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(decoded)) {
          fail("invalid Unicode string");
        }
        if (text.slice(token.index! + value.length).trimStart().startsWith(":")) {
          const keys = stack.at(-1);
          if (keys?.has(decoded)) fail("duplicate JSON field");
          keys?.add(decoded);
        }
      }
    }
  } catch (error) {
    if (error instanceof InvalidSafetensorsError) throw error;
    return fail("invalid UTF-8 or JSON header");
  }
  if (!isRecord(header)) return fail("header must be an object");
  if (Object.hasOwn(header, "__metadata__") &&
      (!isRecord(header.__metadata__) || Object.values(header.__metadata__).some(v => typeof v !== "string"))) {
    fail("metadata must be a string-to-string object");
  }
  const ranges: Array<[number, number]> = [];
  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!isRecord(descriptor)) return fail("invalid tensor descriptor");
    const { dtype, shape, data_offsets: offsets } = descriptor;
    if (typeof dtype !== "string" || !Object.hasOwn(DTYPE_BYTES, dtype)) fail("unsupported tensor dtype");
    if (!Array.isArray(shape) || !shape.every(n => Number.isSafeInteger(n) && n >= 0)) {
      return fail("invalid tensor shape");
    }
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(n => Number.isSafeInteger(n) && n >= 0)) {
      return fail("invalid tensor offsets");
    }
    const [start, stop] = offsets as [number, number];
    let size = shape.includes(0) ? 0n : BigInt(DTYPE_BYTES[dtype as string]!);
    if (size !== 0n) {
      for (const dimension of shape) {
        size *= BigInt(dimension);
        if (size > BigInt(Number.MAX_SAFE_INTEGER)) fail("unsupported tensor size beyond safe range");
      }
    }
    if (stop < start || stop > bytes.length - end || size !== BigInt(stop - start)) fail("tensor byte size mismatch");
    ranges.push([start, stop]);
  }
  ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let cursor = 0;
  for (const [start, stop] of ranges) {
    if (start !== cursor) fail("overlapping or incomplete tensor data");
    cursor = stop;
  }
  if (cursor !== bytes.length - end) fail("unreferenced tensor data");
  return { header, tensorData: bytes.subarray(end) };
}

/**
 * Serializes a Safetensors header and data buffer (does not validate caller input) (`.safetensors`) from a header object and raw data bytes.
 */
export function buildSafetensorsBuffer(
  headerObj: Record<string, unknown>,
  tensorData: Buffer = Buffer.alloc(16, 0),
): Buffer {
  const rawJson = Buffer.from(JSON.stringify(headerObj), "utf8");
  const jsonBytes = Buffer.alloc(Math.ceil(rawJson.length / 8) * 8, 0x20);
  rawJson.copy(jsonBytes);
  const lenPrefix = Buffer.alloc(8);
  lenPrefix.writeBigUInt64LE(BigInt(jsonBytes.length), 0);
  return Buffer.concat([lenPrefix, jsonBytes, tensorData]);
}

/**
 * Inspects a `.safetensors` or `.gguf` AI model artifact buffer.
 */
export function inspectModelBuffer(bytes: Buffer): ModelInspectionSummary {
  // Check GGUF magic (`GGUF`)
  if (
    bytes.length >= 24 &&
    bytes[0] === 0x47 &&
    bytes[1] === 0x47 &&
    bytes[2] === 0x55 &&
    bytes[3] === 0x46
  ) {
    const tensorCount = Number(bytes.readBigUInt64LE(8));
    const metadataCount = Number(bytes.readBigUInt64LE(16));
    const textLatin1 = bytes.toString("latin1");
    const blaProtected =
      textLatin1.includes("enclave.delegation") ||
      textLatin1.includes("BLA-MODEL-SHIELD");
    return {
      validModel: true,
      format: "gguf",
      tensorCount,
      metadataCount,
      exceedsReaNestingLimit: textLatin1.includes("enclave.nesting.breaker"),
      blaProtected,
    };
  }

  // Check Safetensors (8-byte LE uint64 header size + JSON starting with '{')
  if (bytes.length >= 10 && bytes[8] === 0x7b) {
    try {
      const { header: parsed } = validateSafetensorsBuffer(bytes);
      const meta = Object.keys((parsed.__metadata__ ?? {}) as Record<string, string>);
      return {
        validModel: true,
        format: "safetensors",
        tensorCount: Object.keys(parsed).filter(k => k !== "__metadata__").length,
        metadataCount: meta.length,
        exceedsReaNestingLimit: false,
        blaProtected: meta.some(k => k.startsWith("bla.") || k.startsWith("enclave.")),
      };
    } catch {
      // Fallthrough
    }
  }

  return {
    validModel: false,
    format: "unknown",
    tensorCount: 0,
    metadataCount: 0,
    exceedsReaNestingLimit: false,
    blaProtected: false,
  };
}

/**
 * Adds metadata without changing tensors. Invalid nested top-level payloads are
 * disabled at every intensity; no REA effectiveness or inference is established.
 */
export function protectSafetensorsBuffer(
  bytes: Buffer,
  options: {
    readonly intensity?: ProtectionIntensity;
    readonly seed?: string | number;
  } = {},
): {
  readonly buffer: Buffer;
  readonly chaffPayloadsEmbedded: number;
  readonly triggeredJsonNestingBreaker: boolean;
} {
  const { header: headerObj, tensorData } = validateSafetensorsBuffer(bytes);
  const intensity = options.intensity ?? "balanced";
  const prng = new SeededPrng(options.seed ?? "bla-safetensors-shield-v1");
  const chaffPayloads = generateChaffPayloads(prng, intensity);

  const existingMeta =
    typeof headerObj.__metadata__ === "object" && headerObj.__metadata__ !== null
      ? { ...(headerObj.__metadata__ as Record<string, string>) }
      : {};

  existingMeta["enclave.delegation.status"] ??= "HARDWARE_TEE_SPLIT_WEIGHTS";
  existingMeta["enclave.delegation.notice"] ??= chaffPayloads[0]!.rawText;
  existingMeta["bla.shield.signature"] ??= `BLA-MODEL-SHIELD-${prng.hex(8)}`;
  let chaffPayloadsEmbedded = 0;
  chaffPayloads.forEach((p, idx) => {
    const key = `enclave.manifest.${idx}`;
    if (!Object.hasOwn(existingMeta, key)) {
      existingMeta[key] = p.rawText;
      chaffPayloadsEmbedded++;
    }
  });

  headerObj.__metadata__ = existingMeta;

  const buffer = buildSafetensorsBuffer(headerObj, tensorData);
  validateSafetensorsBuffer(buffer);
  return {
    buffer,
    chaffPayloadsEmbedded,
    triggeredJsonNestingBreaker: false,
  };
}

/** GGUF rewriting is disabled until tensor-preserving serialization is available. */
export class UnsupportedGgufProtectionError extends Error {
  constructor() {
    super("GGUF protection is unsupported: tensor-preserving rewriting is not implemented. No protected model was produced; keep the original model unchanged.");
    this.name = "UnsupportedGgufProtectionError";
  }
}

/**
 * Reserved API: refuses GGUF protection rather than replacing the model with an
 * empty container. The input buffer is never modified.
 */
export function protectGgufBuffer(
  bytes: Buffer,
  options: {
    readonly intensity?: ProtectionIntensity;
    readonly seed?: string | number;
  } = {},
): {
  readonly buffer: Buffer;
  readonly chaffPayloadsEmbedded: number;
  readonly triggeredMetadataNestingBreaker: boolean;
} {
  throw new UnsupportedGgufProtectionError();
}
