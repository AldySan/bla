import { afterEach, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildSafetensorsBuffer, inspectModelBuffer, InvalidSafetensorsError, protectSafetensorsBuffer, validateSafetensorsBuffer } from "../src/shields/formats/model.js";
import { protectTarget } from "../src/engine/protector.js";

const header = {
  weights: { dtype: "F32", shape: [2], data_offsets: [0, 8] },
  bias: { dtype: "I16", shape: [2], data_offsets: [8, 12] },
  empty: { dtype: "F32", shape: [0], data_offsets: [12, 12] },
  __metadata__: { description: "trained weights — 测试; 1.0000000000000001 1e999 9007199254740993", "bla.shield.signature": "keep-original", "enclave.manifest.0": "keep-this-too" },
};
const data = Buffer.alloc(12);
data.writeFloatLE(1.25, 0);
data.writeFloatLE(-3.5, 4);
data.writeInt16LE(127, 8);
data.writeInt16LE(-256, 10);
const original = buildSafetensorsBuffer(header, data);

function rawHeader(text: string, tensorData = data): Buffer {
  const json = Buffer.from(text);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([prefix, json, tensorData]);
}

describe("Safetensors format and tensor preservation (not inference)", () => {
  it.each(["stealth", "balanced", "maximum", "bankrupt"] as const)("preserves descriptors, nonzero tensor bytes, and existing metadata at %s", intensity => {
    const source = Buffer.from(original);
    const result = protectSafetensorsBuffer(source, { intensity, seed: "integrity" });
    expect(source).toEqual(original);
    expect(result.triggeredJsonNestingBreaker).toBe(false);
    // Read header and body independently of BLA's inspection/validation helper.
    const length = Number(result.buffer.readBigUInt64LE(0));
    expect(length % 8).toBe(0);
    const parsed = JSON.parse(result.buffer.subarray(8, 8 + length).toString("utf8"));
    expect(Object.keys(parsed).sort()).toEqual(Object.keys(header).sort());
    for (const key of ["weights", "bias", "empty"] as const) expect(parsed[key]).toEqual(header[key]);
    expect(parsed.__metadata__).toMatchObject(header.__metadata__);
    expect(Object.values(parsed.__metadata__).every(v => typeof v === "string")).toBe(true);
    expect(result.chaffPayloadsEmbedded).toBe(Object.keys(parsed.__metadata__).filter(k => k.startsWith("enclave.manifest.") && !Object.hasOwn(header.__metadata__, k)).length);
    expect(result.buffer.subarray(8 + length)).toEqual(data);
    expect(inspectModelBuffer(result.buffer)).toMatchObject({ validModel: true, tensorCount: 3, exceedsReaNestingLimit: false });
  });

  const malformed: Array<[string, Buffer]> = [
    ["empty", Buffer.alloc(0)],
    ["truncated prefix", Buffer.alloc(7)],
    ["oversized length", Buffer.from("ffffffffffffffff7b7d", "hex")],
    ["UTF-8 BOM", rawHeader("\uFEFF{}", Buffer.alloc(0))],
    ["unpaired surrogate", rawHeader(JSON.stringify({ __metadata__: { text: "\uD800" } }), Buffer.alloc(0))],
    ["invalid JSON", rawHeader("{broken")],
    ["non-object", rawHeader("[]")],
    ["non-string metadata", buildSafetensorsBuffer({ ...header, __metadata__: { bad: 1 } }, data)],
    ["array metadata", buildSafetensorsBuffer({ ...header, __metadata__: [] }, data)],
    ["invalid descriptor", buildSafetensorsBuffer({ ...header, _bla_nesting_breaker: [] }, data)],
    ["negative dimension", buildSafetensorsBuffer({ x: { dtype: "F32", shape: [-1], data_offsets: [0, 12] } }, data)],
    ["unsafe dimension", buildSafetensorsBuffer({ x: { dtype: "F32", shape: [Number.MAX_SAFE_INTEGER + 1], data_offsets: [0, 12] } }, data)],
    ["overflowing shape product", buildSafetensorsBuffer({ x: { dtype: "F32", shape: [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER], data_offsets: [0, 12] } }, data)],
    ["size mismatch", buildSafetensorsBuffer({ x: { dtype: "F32", shape: [4], data_offsets: [0, 12] } }, data)],
    ["hole", buildSafetensorsBuffer({ x: { dtype: "F32", shape: [2], data_offsets: [4, 12] } }, data)],
    ["overlap", buildSafetensorsBuffer({ ...header, bias: { dtype: "I16", shape: [2], data_offsets: [4, 8] } }, data)],
    ["unreferenced bytes", buildSafetensorsBuffer({}, data)],
    ["truncated body", original.subarray(0, original.length - 1)],
    ["duplicate name", rawHeader('{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},"x":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}')],
    ["escaped duplicate metadata", rawHeader('{"__metadata__":{"a":"1","\\u0061":"2"}}', Buffer.alloc(0))],
    ["invalid UTF-8", Buffer.from([2, 0, 0, 0, 0, 0, 0, 0, 0x7b, 0xff])],
  ];
  it.each(malformed)("rejects %s instead of manufacturing a model", (_label, bytes) => {
    const snapshot = Buffer.from(bytes);
    expect(() => protectSafetensorsBuffer(bytes)).toThrow(InvalidSafetensorsError);
    expect(inspectModelBuffer(bytes).validModel).toBe(false);
    expect(bytes).toEqual(snapshot);
  });
  it.each(["1.0000000000000001", "1.5", "1e0", "1e-999", "1e999", "9007199254740993", "-0", "9".repeat(128)])(
    "rejects unsupported numeric token %s before rounded values can be accepted", token => {
      const bytes = rawHeader(`{"x":{"dtype":"F32","shape":[${token}],"data_offsets":[0,4]}}`, data.subarray(0, 4));
      expect(() => protectSafetensorsBuffer(bytes)).toThrow(/unsupported numeric token/);
      expect(inspectModelBuffer(bytes).validModel).toBe(false);
    },
  );
  it.each(["4.0000000000000001", "-0"])("rejects unsupported offset token %s", token => {
    const bytes = rawHeader(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[${token === "-0" ? "-0,4" : `0,${token}`}]}}`, data.subarray(0, 4));
    expect(() => protectSafetensorsBuffer(bytes)).toThrow(/unsupported numeric token/);
  });
  it("explicitly reports unsupported dtypes rather than rewriting them", () => {
    const model = buildSafetensorsBuffer({ x: { dtype: "F8_E4M3", shape: [12], data_offsets: [0, 12] } }, data);
    expect(() => protectSafetensorsBuffer(model)).toThrow(/unsupported tensor dtype/);
  });
  it("accepts a scalar and a valid zero-tensor container", () => {
    expect(validateSafetensorsBuffer(buildSafetensorsBuffer({}, Buffer.alloc(0))).tensorData.length).toBe(0);
    expect(validateSafetensorsBuffer(buildSafetensorsBuffer({ x: { dtype: "F32", shape: [], data_offsets: [0, 4] } }, data.subarray(0, 4))).tensorData).toEqual(data.subarray(0, 4));
  });
});

describe("Safetensors malformed input no-write admission", () => {
  let dir: string | undefined;
  afterEach(() => { if (dir) rmSync(dir, { recursive: true, force: true }); });
  it("does not create output directories or overwrite input/existing output", () => {
    dir = mkdtempSync(join(tmpdir(), "bla-safetensors-integrity-"));
    const input = join(dir, "broken.safetensors");
    const output = join(dir, "existing.safetensors");
    const bad = Buffer.from("invalid-model");
    writeFileSync(input, bad);
    writeFileSync(output, original);
    for (const options of [
      { inputPath: input }, { inputPath: input, inPlace: true },
      { inputPath: input, outputPath: input }, { inputPath: input, outputPath: output },
      { inputPath: input, outputPath: join(dir, "new", "out.safetensors") },
    ]) {
      if (options.outputPath === input && !options.inPlace) {
        expect(() => protectTarget(options)).toThrow(/Output refers to the input file/);
      } else {
        expect(() => protectTarget(options)).toThrow(InvalidSafetensorsError);
      }
    }
    expect(readFileSync(input)).toEqual(bad);
    expect(readFileSync(output)).toEqual(original);
    expect(existsSync(join(dir, "new"))).toBe(false);
    expect(existsSync(join(dir, "broken.shielded.safetensors"))).toBe(false);
  });
});
