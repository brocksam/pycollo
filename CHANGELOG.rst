*****************
Pycollo Changelog
*****************

:Info: Changelog for Pycollo releases.
:Authors: Sam Brockie (sambrockie@icloud.com)
:Date: 2021-06-17
:Version: 0.2.0

GitHub holds releases, too
==========================

More information can be found on GitHub in the `releases section
<https://github.com/brocksam/pycollo/releases>`_.

About this Changelog
====================

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates should be (year-month-day) to conform with [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html).

Formatting a New Version
========================

Include sections:

- Added - for new features.
- Changed - for changes in existing functionality.
- Depricated - for soon-to-be removed features.
- Removed - for now removed features.
- Fixed - for any bug fixes.
- Security - in case of vulnerabilities.

Version History
===============

Unreleased
----------

- None

[0.2.0] - 2021-06-17
--------------------

Changes
~~~~~~~

- Changed the default derivative backend to CasADi [PR `#29`_].
- Moved CI to GitHub Actions [PR `#36`_].
- Modernise packaging based on PEP 517/PEP 518 using a ``setup.cfg`` and ``pyproject.toml``-based approach [PR `#38`_].

.. _#29: https://github.com/brocksam/pycollo/pull/29
.. _#36: https://github.com/brocksam/pycollo/pull/36
.. _#38: https://github.com/brocksam/pycollo/pull/38
