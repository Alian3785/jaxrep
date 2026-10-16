"""Read native SG records from installed loose files or Steam VPK archives.

Offline source import only. This module never starts Disciples, JAX, or a policy.
"""
from pathlib import Path
import re
import struct
import zlib


class Record:
    def __init__(self, kind, data):
        self.kind = kind
        self.data = data
        self.id = self.text('OBJ_ID')

    def blob(self, field):
        offset = self.data.index(field.encode('ascii')) + len(field)
        size = struct.unpack_from('<I', self.data, offset)[0]
        if offset + 4 + size > len(self.data):
            raise ValueError(f'{self.id if hasattr(self, "id") else self.kind}: invalid {field}')
        return self.data[offset + 4:offset + 4 + size]

    def text(self, field):
        return self.blob(field).rstrip(b'\0').decode('cp1252')

    def integer(self, field):
        return struct.unpack_from('<i', self.data, self.data.index(field.encode('ascii')) + len(field))[0]

    def ids(self, field):
        pattern = re.escape(field.encode()) + rb'\x0b\x00\x00\x00([SGC][A-Za-z0-9]{9})\x00'
        return [m[1].decode().upper() for m in re.finditer(pattern, self.data)]

    @property
    def position(self):
        return [self.integer('POS_X'), self.integer('POS_Y')]


def records(data):
    """Typed record headers delimit SG objects; validate each class-string length."""
    result = []
    for match in re.finditer(rb'WHAT(.{4})(\.\?AV[^\x00]+)\x00', data, re.DOTALL):
        if struct.unpack('<I', match[1])[0] != len(match[2]) + 1:
            raise ValueError('Invalid SG object class length')
        end = data.index(b'ENDOBJECT\x00', match.end()) + len(b'ENDOBJECT\x00')
        result.append(Record(match[2].decode(), data[match.start():end]))
    if not result:
        raise ValueError('No native scenario records found')
    if len({row.id for row in result}) != len(result):
        raise ValueError('Duplicate native scenario object IDs')
    return result


def read_scenario(game, relative_path):
    """Return exact source bytes plus portable provenance, verifying VPK CRC32."""
    game = Path(game)
    relative_path = relative_path.replace('\\', '/').lower()
    loose = next((p for p in game.rglob('*.sg') if p.relative_to(game).as_posix().lower() == relative_path), None)
    if loose is not None:
        return loose.read_bytes(), dict(file=loose.relative_to(game).as_posix())
    for index in sorted((game / 'vpks').glob('*_dir.vpk')):
        directory = index.read_bytes()
        magic, version, tree_size = struct.unpack_from('<III', directory)
        if magic != 0x55aa1234 or version not in (1, 2):
            raise ValueError('Unsupported Steam VPK header')
        header_size = 12 if version == 1 else 28
        cursor = header_size

        def cstring(directory=directory):
            nonlocal cursor
            end = directory.index(0, cursor)
            value = directory[cursor:end].decode('utf-8')
            cursor = end + 1
            return value

        while (extension := cstring()):
            while (folder := cstring()):
                while (name := cstring()):
                    crc, preload_size, archive, offset, length, terminator = struct.unpack_from('<IHHIIH', directory, cursor)
                    cursor += 18
                    preload = directory[cursor:cursor + preload_size]
                    cursor += preload_size
                    if terminator != 0xffff:
                        raise ValueError('Invalid VPK tree entry')
                    path = f'{folder}/{name}.{extension}'.lower()
                    if path != relative_path:
                        continue
                    source = index if archive == 0x7fff else index.with_name(index.name.replace('_dir.vpk', f'_{archive:03d}.vpk'))
                    with source.open('rb') as stream:
                        stream.seek(offset + (header_size + tree_size if archive == 0x7fff else 0))
                        data = preload + stream.read(length)
                    if zlib.crc32(data) != crc:
                        raise ValueError('VPK scenario CRC32 mismatch')
                    return data, dict(index=index.relative_to(game).as_posix(), archive=source.relative_to(game).as_posix(),
                                      file=path, offset=offset, length=length, crc32=f'{crc:08x}')
    raise FileNotFoundError(relative_path)
