"""Fixed-schema numeric model transport. Never unpickle network input."""
import hashlib
import io
import zipfile
import numpy as np
import torch
from ..models import ThreeStream

PROTOCOL = 1
MAX_BODY = 2_000_000


def digest(body):
    return hashlib.sha256(body).hexdigest()


def encode(state):
    sink = io.BytesIO()
    np.savez(sink, **{k: v.detach().cpu().numpy() for k, v in state.items()})
    return sink.getvalue()


def decode(body, template=None):
    """Accept only finite float32 tensors matching this project's ThreeStream."""
    template = ThreeStream().state_dict() if template is None else template
    if len(body) > MAX_BODY:
        raise ValueError('Model body is too large')
    expected = {k+'.npy' for k in template}
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {m.filename for m in members} != expected:
                raise ValueError('Unexpected or duplicate model parameters')
            limit = sum(v.numel()*v.element_size() for v in template.values()) + 1024*len(template)
            if sum(m.file_size for m in members) > limit:
                raise ValueError('Expanded model body is too large')
            # Validate shapes before np.load can allocate from an untrusted header.
            for key, target in template.items():
                with archive.open(key+'.npy') as member:
                    version = np.lib.format.read_magic(member)
                    if version != (1, 0):
                        raise ValueError('Unsupported parameter header version')
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=1024)
                    if shape != tuple(target.shape) or dtype != np.dtype('float32') or fortran:
                        raise ValueError(f'Parameter shape/dtype mismatch: {key}')
        with np.load(io.BytesIO(body), allow_pickle=False, max_header_size=1024) as source:
            result = {}
            for key, target in template.items():
                value = source[key]
                if value.shape != tuple(target.shape) or value.dtype != np.dtype('float32'):
                    raise ValueError(f'Parameter shape/dtype mismatch: {key}')
                if not np.isfinite(value).all():
                    raise ValueError(f'Non-finite parameter: {key}')
                result[key] = torch.from_numpy(value.copy())
            return result
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as exc:
        raise ValueError('Invalid numeric model archive') from exc
