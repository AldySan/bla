import { afterEach, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { protectZipOrApkBuffer, readZipEntries, UnsupportedApkProtectionError, UnsupportedArchiveError, writeZipArchive } from "../src/shields/formats/archive.js";
import { protectTarget } from "../src/engine/protector.js";

const fixture = () => writeZipArchive([{ name: "file.txt", data: Buffer.from("harmless content ".repeat(8)) }]);
const centralOffset = (b: Buffer) => b.readUInt32LE(b.length - 6);
function mutate(change: (b: Buffer, c: number) => void): Buffer {
  const b = fixture(); change(b, centralOffset(b)); return b;
}
function gapBeforeDirectory(asCompressedData: boolean): Buffer {
  const b = fixture(); const c = centralOffset(b);
  const result = Buffer.concat([b.subarray(0, c), Buffer.from([0]), b.subarray(c)]);
  result.writeUInt32LE(c + 1, result.length - 6);
  if (asCompressedData) {
    result.writeUInt32LE(b.readUInt32LE(18) + 1, 18);
    result.writeUInt32LE(b.readUInt32LE(18) + 1, c + 1 + 20);
  }
  return result;
}
const bad: [string, () => Buffer][] = [
  ["unrepresented signing-block-like gap", () => gapBeforeDirectory(false)],
  ["trailing compressed garbage", () => gapBeforeDirectory(true)],
  ["extra fields", () => mutate((b, c) => { b.writeUInt16LE(1, c + 30); })],
  ["central/local mismatch", () => mutate((b, c) => { b[c + 46] = 120; })],
  ["trailing archive bytes", () => Buffer.concat([fixture(), Buffer.from([0])])],
  ["data descriptors", () => mutate((b, c) => { b.writeUInt16LE(8, 6); b.writeUInt16LE(8, c + 8); })],
  ["encryption", () => mutate((b, c) => { b.writeUInt16LE(1, 6); b.writeUInt16LE(1, c + 8); })],
  ["unknown compression", () => mutate((b, c) => { b.writeUInt16LE(99, 8); b.writeUInt16LE(99, c + 10); })],
  ["truncated EOCD", () => fixture().subarray(0, -1)],
  ["bad CRC", () => mutate((b, c) => { b.writeUInt32LE(0, 14); b.writeUInt32LE(0, c + 16); })],
  ["invalid deflate", () => mutate((b) => { b[38] = 0xff; })],
  ["decoded size mismatch", () => mutate((b, c) => { b.writeUInt32LE(1, 22); b.writeUInt32LE(1, c + 24); })],
  ["oversized declared output", () => mutate((b, c) => { b.writeUInt32LE(64 * 1024 * 1024 + 1, 22); b.writeUInt32LE(64 * 1024 * 1024 + 1, c + 24); })],
  ["missing entries", () => mutate((b) => { b.writeUInt16LE(2, b.length - 14); b.writeUInt16LE(2, b.length - 12); })],
  ["incorrect local offset", () => mutate((b, c) => { b.writeUInt32LE(1, c + 42); })],
  ["unpreserved timestamps", () => mutate((b, c) => { b.writeUInt16LE(33, 12); b.writeUInt16LE(33, c + 14); })],
  ["unpreserved attributes", () => mutate((b, c) => { b.writeUInt32LE(0x81a40000, c + 38); })],
  ["truncated entry", () => mutate((b, c) => { b.writeUInt32LE(c, 18); b.writeUInt32LE(c, c + 20); })],
  ["ZIP64", () => mutate((b) => { b.writeUInt16LE(0xffff, b.length - 12); })],
];

describe("archive admission preserves bytes or fails closed", () => {
  const dirs: string[] = [];
  afterEach(() => { for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true }); });
  it("reads an independently specified classic stored ZIP fixture", () => {
    const input = Buffer.from([
      "504b03041400000000000000000086a610360500000005000000010000006168656c6c6f",
      "504b010214001400000000000000000086a61036050000000500000001000000000000000000000000000000000061",
      "504b050600000000010001002f000000240000000000",
    ].join(""), "hex");
    expect(readZipEntries(input)[0]?.data.toString()).toBe("hello");
  });
  it("preserves stored and deflated file contents", () => {
    const originals = [{ name: "short", data: Buffer.from("hello") }, { name: "long", data: Buffer.from("plain content ".repeat(20)) }];
    const input = writeZipArchive(originals);
    const result = protectZipOrApkBuffer(input, { intensity: "stealth", seed: "integrity" });
    const entries = readZipEntries(result.buffer);
    for (const item of originals) expect(entries.find((e) => e.name === item.name)?.data).toEqual(item.data);
  });
  it.each(bad)("rejects %s without changing the supplied buffer", (_name, create) => {
    const input = create(); const before = Buffer.from(input);
    expect(() => readZipEntries(input)).toThrow(UnsupportedArchiveError);
    expect(() => protectZipOrApkBuffer(input)).toThrow(UnsupportedArchiveError);
    expect(input).toEqual(before);
  });
  it("refuses unsigned APKs and APK contents renamed as ZIP", () => {
    expect(() => protectZipOrApkBuffer(fixture(), { isApk: true })).toThrow(UnsupportedApkProtectionError);
    const renamed = writeZipArchive([{ name: "AndroidManifest.xml", data: Buffer.from("fixture") }]);
    expect(() => protectZipOrApkBuffer(renamed)).toThrow(/signing-aware/);
  });
  it("refuses signed archives and duplicate names", () => {
    expect(() => protectZipOrApkBuffer(writeZipArchive([{ name: "META-INF/TEST.RSA", data: Buffer.from("not a real signature") }]))).toThrow(/signing-aware/);
    expect(() => readZipEntries(writeZipArchive([{ name: "a", data: Buffer.from("1") }, { name: "a", data: Buffer.from("2") }]))).toThrow(/duplicate/);
  });
  it.each(["bad.zip", "unsigned.apk", "renamed.zip"])("retains input/output and creates no directory for %s", (name) => {
    const dir = mkdtempSync(join(tmpdir(), "bla-archive-")); dirs.push(dir);
    const input = join(dir, name); const output = join(dir, "existing.zip");
    const bytes = name === "bad.zip" ? bad[0]![1]() : name === "renamed.zip" ? writeZipArchive([{ name: "AndroidManifest.xml", data: Buffer.from("fixture") }]) : fixture();
    writeFileSync(input, bytes); writeFileSync(output, "existing output");
    expect(() => protectTarget({ inputPath: input, outputPath: output })).toThrow();
    expect(readFileSync(output, "utf8")).toBe("existing output");
    expect(() => protectTarget({ inputPath: input, inPlace: true })).toThrow();
    expect(readFileSync(input)).toEqual(bytes);
    expect(() => protectTarget({ inputPath: input, outputPath: join(dir, "new", "out.zip") })).toThrow();
    expect(existsSync(join(dir, "new"))).toBe(false);
  });
});
