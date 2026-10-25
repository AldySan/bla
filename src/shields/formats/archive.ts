import { deflateRawSync, inflateRawSync } from "node:zlib";
import { SeededPrng } from "../../core/prng.js";
import type { ProtectionIntensity } from "../../core/types.js";
import { generateChaffPayloads } from "../chaff/chaff-generator.js";

export interface ZipEntryInfo {
  readonly name: string;
  readonly compressedSize: number;
  readonly uncompressedSize: number;
  readonly compressionMethod: number;
  readonly data: Buffer;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc = CRC32_TABLE[(crc ^ buf[i]!) & 0xff]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export class UnsupportedArchiveError extends Error {
  constructor(reason: string) {
    super(`ZIP input cannot be rewritten safely: ${reason}`);
    this.name = "UnsupportedArchiveError";
  }
}

export class UnsupportedApkProtectionError extends Error {
  constructor() {
    super("APK protection is unsupported: a signing-aware, integrity-preserving pipeline is required. Input and existing output are unchanged.");
    this.name = "UnsupportedApkProtectionError";
  }
}

/** Reads only the explicitly supported, metadata-free classic ZIP subset. */
export function readZipEntries(bytes: Buffer): readonly ZipEntryInfo[] {
  const reject = (reason: string): never => { throw new UnsupportedArchiveError(reason); };
  const end = bytes.length - 22;
  if (end < 0 || bytes.readUInt32LE(end) !== 0x06054b50) reject("missing terminal EOCD, comments, or trailing data");
  if (bytes.readUInt16LE(end + 4) || bytes.readUInt16LE(end + 6) || bytes.readUInt16LE(end + 20)) reject("multi-disk archives or comments");
  const count = bytes.readUInt16LE(end + 10);
  const size = bytes.readUInt32LE(end + 12);
  const start = bytes.readUInt32LE(end + 16);
  if (count === 0xffff || size === 0xffffffff || start === 0xffffffff) reject("ZIP64");
  if (bytes.readUInt16LE(end + 8) !== count || start + size !== end) reject("inconsistent central directory");
  const entries: ZipEntryInfo[] = [];
  const names = new Set<string>();
  let local = 0;
  let central = start;
  let decodedTotal = 0;
  for (let i = 0; i < count; i++) {
    if (central + 46 > end || bytes.readUInt32LE(central) !== 0x02014b50) reject("truncated central directory");
    const flags = bytes.readUInt16LE(central + 8);
    const method = bytes.readUInt16LE(central + 10);
    const crc = bytes.readUInt32LE(central + 16);
    const compressedSize = bytes.readUInt32LE(central + 20);
    const uncompressedSize = bytes.readUInt32LE(central + 24);
    const nameLen = bytes.readUInt16LE(central + 28);
    if (flags !== 0) reject("flags (including encryption, data descriptors, or alternate filename encodings)");
    if (method !== 0 && method !== 8) reject("unsupported compression method");
    if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff) reject("ZIP64 sizes");
    if (bytes.readUInt16LE(central + 6) > 20 || bytes.readUInt32LE(central + 12) ||
        bytes.readUInt16LE(central + 30) || bytes.readUInt16LE(central + 32) ||
        bytes.readUInt16LE(central + 34) || bytes.readUInt16LE(central + 36) || bytes.readUInt32LE(central + 38)) {
      reject("unsupported versions, timestamps, extra fields, comments, disks, or file attributes");
    }
    if (central + 46 + nameLen > end || bytes.readUInt32LE(central + 42) !== local) reject("entry offsets or names are inconsistent");
    if (local + 30 > start || bytes.readUInt32LE(local) !== 0x04034b50) reject("missing local header or unrepresented data");
    if (bytes.readUInt16LE(local + 4) !== bytes.readUInt16LE(central + 6) ||
        bytes.readUInt16LE(local + 6) !== flags || bytes.readUInt16LE(local + 8) !== method ||
        bytes.readUInt32LE(local + 10) || bytes.readUInt32LE(local + 14) !== crc ||
        bytes.readUInt32LE(local + 18) !== compressedSize || bytes.readUInt32LE(local + 22) !== uncompressedSize ||
        bytes.readUInt16LE(local + 26) !== nameLen || bytes.readUInt16LE(local + 28)) reject("local/central header mismatch or local metadata");
    const dataStart = local + 30 + nameLen;
    const dataEnd = dataStart + compressedSize;
    if (dataEnd > start) reject("truncated entry data");
    const nameBytes = bytes.subarray(local + 30, dataStart);
    if (!nameBytes.equals(bytes.subarray(central + 46, central + 46 + nameLen)) ||
        nameBytes.length === 0 || nameBytes.some((v) => v < 32 || v > 126)) reject("unsupported or inconsistent filenames");
    const name = nameBytes.toString("ascii");
    if (names.has(name)) reject("duplicate filenames");
    names.add(name);
    decodedTotal += uncompressedSize;
    if (uncompressedSize > 64 * 1024 * 1024 || decodedTotal > 256 * 1024 * 1024) reject("decoded size exceeds admission limits");
    const compressed = bytes.subarray(dataStart, dataEnd);
    let data: Buffer;
    if (method === 0) {
      if (compressedSize !== uncompressedSize) reject("stored entry size mismatch");
      data = Buffer.from(compressed);
    } else {
      try {
        const result = inflateRawSync(compressed, { maxOutputLength: Math.max(1, uncompressedSize + 1), info: true });
        // Node's info result reports consumed input, detecting trailing compressed garbage.
        const decoded = result as unknown as { buffer: Buffer; engine: { bytesWritten: number } };
        if (decoded.engine.bytesWritten !== compressedSize) reject("unconsumed compressed bytes");
        data = decoded.buffer;
      } catch {
        data = reject("invalid or incomplete deflate stream");
      }
    }
    if (data.length !== uncompressedSize || crc32(data) !== crc) reject("decoded size or CRC mismatch");
    entries.push({ name, compressedSize, uncompressedSize, compressionMethod: method, data });
    local = dataEnd;
    central += 46 + nameLen;
  }
  if (central !== end || local !== start) reject("unrepresented entries, signing structures, or gaps");
  return entries;
}

/**
 * Writes a minimal ZIP archive from a list of named file buffers.
 */
export function writeZipArchive(
  files: readonly { readonly name: string; readonly data: Buffer; readonly storeOnly?: boolean }[],
): Buffer {
  const localHeaders: Buffer[] = [];
  const centralHeaders: Buffer[] = [];
  let localOffset = 0;

  for (const file of files) {
    const nameBuf = Buffer.from(file.name, "utf8");
    const uncompressedSize = file.data.length;
    const crc = crc32(file.data);
    const useStore = file.storeOnly === true || uncompressedSize < 64;
    const compressedData = useStore ? file.data : deflateRawSync(file.data);
    const method = useStore ? 0 : 8;

    const local = Buffer.alloc(30 + nameBuf.length);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4); // version needed
    local.writeUInt16LE(0, 6); // flags
    local.writeUInt16LE(method, 8);
    local.writeUInt16LE(0, 10); // mod time
    local.writeUInt16LE(0, 12); // mod date
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(compressedData.length, 18);
    local.writeUInt32LE(uncompressedSize, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28);
    nameBuf.copy(local, 30);

    localHeaders.push(local, compressedData);

    const central = Buffer.alloc(46 + nameBuf.length);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 8);
    central.writeUInt16LE(method, 10);
    central.writeUInt16LE(0, 12);
    central.writeUInt16LE(0, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(compressedData.length, 20);
    central.writeUInt32LE(uncompressedSize, 24);
    central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt16LE(0, 30);
    central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34);
    central.writeUInt16LE(0, 36);
    central.writeUInt32LE(0, 38);
    central.writeUInt32LE(localOffset, 42);
    nameBuf.copy(central, 46);

    centralHeaders.push(central);
    localOffset += local.length + compressedData.length;
  }

  const centralSize = centralHeaders.reduce((acc, b) => acc + b.length, 0);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(0, 4);
  eocd.writeUInt16LE(0, 6);
  eocd.writeUInt16LE(files.length, 8);
  eocd.writeUInt16LE(files.length, 10);
  eocd.writeUInt32LE(centralSize, 12);
  eocd.writeUInt32LE(localOffset, 16);
  eocd.writeUInt16LE(0, 20);

  return Buffer.concat([...localHeaders, ...centralHeaders, eocd]);
}

/** Admission shared with the file engine, before any output paths are created. */
export function validateZipProtectionInput(bytes: Buffer): readonly ZipEntryInfo[] {
  const existingEntries = readZipEntries(bytes);
  if (existingEntries.some((e) => e.name === "AndroidManifest.xml" || /^META-INF\/.*\.(SF|RSA|DSA|EC)$/i.test(e.name))) {
    throw new UnsupportedArchiveError("APK or signed archive requires a signing-aware pipeline");
  }
  if (existingEntries.some((e) => e.name.startsWith("assets/enclave/") || e.name.startsWith("enclave/"))) {
    throw new UnsupportedArchiveError("existing entries overlap the reserved output namespace");
  }
  return existingEntries;
}

/**
 * Rewrites the admitted ZIP subset. APK rewriting is explicitly unsupported.
 * Embedded executable assets are copied, not recursively transformed.
 */
export function protectZipOrApkBuffer(
  bytes: Buffer,
  options: {
    readonly intensity?: ProtectionIntensity;
    readonly seed?: string | number;
    readonly isApk?: boolean;
  } = {},
): {
  readonly buffer: Buffer;
  readonly decoyEntriesInjected: number;
  readonly chaffPayloadsEmbedded: number;
} {
  if (options.isApk) throw new UnsupportedApkProtectionError();
  const existingEntries = validateZipProtectionInput(bytes);
  const intensity = options.intensity ?? "balanced";
  const prng = new SeededPrng(options.seed ?? "bla-archive-shield-v1");
  const chaffPayloads = generateChaffPayloads(prng, intensity);
  const outputFiles: { name: string; data: Buffer; storeOnly?: boolean }[] = existingEntries.map((e) => ({
    name: e.name,
    data: e.data,
  }));

  // Inject decoy enclave asset manifests & shadow class descriptors
  const manifestContent = JSON.stringify(
    {
      schema: "bla-archive-enclave-shield/v1",
      protected: true,
      enclave_delegation: "hardware_tee_ring0",
      manifests: chaffPayloads.map((p) => p.rawText),
    },
    null,
    2,
  );

  outputFiles.push({
    name: "assets/enclave/bla-attestation-manifest.json",
    data: Buffer.from(manifestContent, "utf8"),
  });

  const decoyCount = intensity === "stealth" ? 2 : intensity === "balanced" ? 4 : 8;
  for (let i = 0; i < decoyCount; i++) {
    const hex = prng.hex(6);
    const payload = chaffPayloads[i % chaffPayloads.length]!;
    outputFiles.push({
      name: options.isApk
        ? `assets/enclave/tee_bridge_stub_${hex}.properties`
        : `enclave/shadow_module_${hex}.json`,
      data: Buffer.from(`# ${payload.rawText}\nenclave.cid=${16 + i}\nenclave.sealed=true\n`, "utf8"),
    });
  }

  return {
    buffer: writeZipArchive(outputFiles),
    decoyEntriesInjected: decoyCount + 1,
    chaffPayloadsEmbedded: chaffPayloads.length,
  };
}
