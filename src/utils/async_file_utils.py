"""
Async file utilities to replace blocking file operations
"""

import json
import asyncio
from typing import Any, Dict, Union
from pathlib import Path


async def async_mkdir(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
    """Async directory creation"""
    path = Path(path)
    
    def _mkdir():
        path.mkdir(parents=parents, exist_ok=exist_ok)
    
    await asyncio.to_thread(_mkdir)


async def async_read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Async read text file"""
    file_path = Path(file_path)
    
    def _read():
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    return await asyncio.to_thread(_read)


async def async_write_text(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """Async write text file"""
    file_path = Path(file_path)
    await async_mkdir(file_path.parent)
    
    def _write():
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    await asyncio.to_thread(_write)


async def async_read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """Async read JSON file"""
    content = await async_read_text(file_path, encoding)
    return json.loads(content)


async def async_write_json(file_path: Union[str, Path], data: Any, encoding: str = 'utf-8', indent: int = 2) -> None:
    """Async write JSON file"""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    await async_write_text(file_path, content, encoding)


async def async_read_binary(file_path: Union[str, Path]) -> bytes:
    """Async read binary file"""
    file_path = Path(file_path)
    
    def _read():
        with open(file_path, 'rb') as f:
            return f.read()
    
    return await asyncio.to_thread(_read)


async def async_write_binary(file_path: Union[str, Path], data: bytes) -> None:
    """Async write binary file"""
    file_path = Path(file_path)
    await async_mkdir(file_path.parent)
    
    def _write():
        with open(file_path, 'wb') as f:
            f.write(data)
    
    await asyncio.to_thread(_write)


async def async_append_text(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """Async append to text file"""
    file_path = Path(file_path)
    await async_mkdir(file_path.parent)
    
    def _append():
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
    
    await asyncio.to_thread(_append)


async def async_write_jsonl(file_path: Union[str, Path], items: list, encoding: str = 'utf-8') -> None:
    """Async write JSONL file (one JSON object per line)"""
    lines = []
    for item in items:
        lines.append(json.dumps(item, ensure_ascii=False))
    
    content = '\n'.join(lines) + '\n'
    await async_write_text(file_path, content, encoding)


async def async_append_jsonl(file_path: Union[str, Path], item: Any, encoding: str = 'utf-8') -> None:
    """Async append single JSON object to JSONL file"""
    content = json.dumps(item, ensure_ascii=False) + '\n'
    await async_append_text(file_path, content, encoding) 