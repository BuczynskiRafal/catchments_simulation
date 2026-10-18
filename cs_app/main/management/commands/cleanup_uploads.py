"""Management command to purge stale uploaded SWMM files from MEDIA_ROOT/uploaded_files."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Remove uploaded SWMM files older than --max-age-hours from MEDIA_ROOT/uploaded_files."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--max-age-hours",
            type=float,
            default=24.0,
            help="Delete uploaded files last modified more than this many hours ago (default: 24).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List files that would be removed without deleting them.",
        )

    def handle(self, *args: object, **options: object) -> str:
        max_age_hours = float(options["max_age_hours"])
        if max_age_hours < 0:
            raise CommandError("--max-age-hours must be non-negative.")

        dry_run = bool(options["dry_run"])
        cutoff = time.time() - (max_age_hours * 3600.0)
        upload_root = os.path.realpath(os.path.join(settings.MEDIA_ROOT, "uploaded_files"))

        if not os.path.isdir(upload_root):
            message = f"Upload directory does not exist: {upload_root} (0 files removed)."
            self.stdout.write(message)
            return message

        removed_count = 0
        for dirpath, _dirnames, filenames in os.walk(upload_root, topdown=False):
            real_dirpath = os.path.realpath(dirpath)
            if os.path.commonpath([upload_root, real_dirpath]) != upload_root:
                continue

            for filename in filenames:
                filepath = os.path.join(real_dirpath, filename)
                if os.path.islink(filepath):
                    continue
                try:
                    stat_result = os.stat(filepath)
                except OSError:
                    continue

                if stat_result.st_mtime <= cutoff:
                    if not dry_run:
                        with suppress(OSError):
                            os.remove(filepath)
                            removed_count += 1
                    else:
                        removed_count += 1

            if not dry_run and real_dirpath != upload_root:
                with suppress(OSError):
                    if not os.listdir(real_dirpath):
                        os.rmdir(real_dirpath)

        action = "Would remove" if dry_run else "Removed"
        message = f"{action} {removed_count} expired upload(s) from {upload_root}."
        self.stdout.write(message)
        return message
