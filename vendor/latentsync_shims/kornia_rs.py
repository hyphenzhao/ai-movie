"""stub: kornia only needs kornia_rs for image file I/O, which LatentSync never calls."""
def __getattr__(name):
    raise AttributeError(f"kornia_rs stub has no {name}")
