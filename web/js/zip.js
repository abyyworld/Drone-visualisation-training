/**
 * A minimal ZIP writer, store-only.
 *
 * WHY NOT A LIBRARY
 *     The one thing this needs to do is put a handful of JSON files and the images they
 *     describe into one download. Everything going in is already compressed - JPEG, PNG,
 *     MP4 - so deflate would spend time to save nothing, and "store" is a length and a
 *     checksum. That is ninety lines. A dependency for it would be a megabyte on every
 *     page load and one more thing that can go stale.
 *
 * WHAT IT DELIBERATELY DOES NOT DO
 *     Compression, encryption, or Zip64. Zip64 is the real limit: this cannot write an
 *     archive over 4 GB or a member over 4 GB, and it says so rather than writing something
 *     an unzip tool will refuse halfway through.
 */

const LOCAL_HEADER = 0x04034b50;
const CENTRAL_HEADER = 0x02014b50;
const END_OF_CENTRAL = 0x06054b50;
const MAX_SIZE = 0xffffffff;

/** CRC-32, the checksum a ZIP member carries. Table built once on first use. */
let crcTable = null;

function crc32(bytes) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let i = 0; i < 256; i += 1) {
      let value = i;
      for (let bit = 0; bit < 8; bit += 1) {
        value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
      }
      crcTable[i] = value >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** MS-DOS date and time, which is what a ZIP header stores. Two seconds of resolution. */
function dosDateTime(date) {
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const day = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { time, day };
}

/**
 * Build a ZIP.
 *
 * @param {Array<{name: string, data: Uint8Array}>} entries
 * @returns {Blob}
 */
export function zip(entries) {
  const encoder = new TextEncoder();
  const now = dosDateTime(new Date());
  const parts = [];
  const central = [];
  let offset = 0;

  for (const entry of entries) {
    const name = encoder.encode(entry.name);
    const data = entry.data;
    if (data.length > MAX_SIZE) {
      throw new Error(`${entry.name} is too large for a ZIP without Zip64 support.`);
    }
    const checksum = crc32(data);

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, LOCAL_HEADER, true);
    local.setUint16(4, 20, true);          // version needed
    local.setUint16(6, 0x0800, true);      // UTF-8 names
    local.setUint16(8, 0, true);           // stored, not deflated
    local.setUint16(10, now.time, true);
    local.setUint16(12, now.day, true);
    local.setUint32(14, checksum, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);          // no extra field

    parts.push(new Uint8Array(local.buffer), name, data);

    const entryHeader = new DataView(new ArrayBuffer(46));
    entryHeader.setUint32(0, CENTRAL_HEADER, true);
    entryHeader.setUint16(4, 20, true);    // version made by
    entryHeader.setUint16(6, 20, true);    // version needed
    entryHeader.setUint16(8, 0x0800, true);
    entryHeader.setUint16(10, 0, true);
    entryHeader.setUint16(12, now.time, true);
    entryHeader.setUint16(14, now.day, true);
    entryHeader.setUint32(16, checksum, true);
    entryHeader.setUint32(20, data.length, true);
    entryHeader.setUint32(24, data.length, true);
    entryHeader.setUint16(28, name.length, true);
    entryHeader.setUint32(42, offset, true);
    central.push(new Uint8Array(entryHeader.buffer), name);

    offset += 30 + name.length + data.length;
    if (offset > MAX_SIZE) {
      throw new Error('The archive is too large for a ZIP without Zip64 support.');
    }
  }

  const centralSize = central.reduce((total, part) => total + part.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, END_OF_CENTRAL, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);

  return new Blob([...parts, ...central, new Uint8Array(end.buffer)],
    { type: 'application/zip' });
}
