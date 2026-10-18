# Security policy

## Supported versions

The package is in the 0.x series. Security fixes are made only in the latest
release published on [PyPI](https://pypi.org/project/catchment-simulation/);
older 0.x releases do not receive fixes, so please upgrade before reporting.

| Version              | Supported |
|----------------------|-----------|
| latest 0.x release   | yes       |
| older releases       | no        |

The Django application in `cs_app/` is not released as a package. Reports
about it, and about the public instance at
<https://catchment-simulations.onrender.com/>, are welcome through the same
channel.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub's private vulnerability reporting:
open the repository's **Security** tab and choose **Report a vulnerability**,
or go directly to
<https://github.com/BuczynskiRafal/catchments_simulation/security/advisories/new>.
If that form is not available, email the maintainer at
<rafal.buczynski@sd.pb.edu.pl> instead.

Include what you can of:

- the affected version and how you installed it,
- a description of the problem and its impact,
- steps or a minimal SWMM model that reproduce it.

## What to expect

This is a project maintained by one person in their spare time, so the
timelines below are goals rather than guarantees:

- an acknowledgement within 7 days,
- an assessment, and a plan or a request for more details, within 30 days,
- a fix released as a new version, announced in the
  [changelog](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/CHANGELOG.md)
  and in a GitHub Security Advisory that credits you unless you prefer
  otherwise.

Please keep the report confidential until a fix is released.
