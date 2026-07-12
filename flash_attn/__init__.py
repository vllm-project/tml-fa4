from pkgutil import extend_path

# Allow this FA4 forward subset to coexist with a full flash_attn install.
__path__ = extend_path(__path__, __name__)
