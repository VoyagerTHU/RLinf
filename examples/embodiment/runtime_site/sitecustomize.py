"""Append optional dependency environments after the primary interpreter site."""

from __future__ import annotations

import os
import sys

for extra_site in os.environ.get("RLINF_EXTRA_SITE_PACKAGES", "").split(os.pathsep):
    extra_site = extra_site.strip()
    if extra_site and os.path.isdir(extra_site) and extra_site not in sys.path:
        sys.path.append(extra_site)

