<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# Developer Certificate of Origin

Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

---

## Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

**(a)** The contribution was created in whole or in part by me and I
have the right to submit it under the open source license
indicated in the file; or

**(b)** The contribution is based upon previous work that, to the best
of my knowledge, is covered under an appropriate open source
license and I have the right under that license to submit that
work with modifications, whether created in whole or in part
by me, under the same open source license (unless I am
permitted to submit under a different license), as indicated
in the file; or

**(c)** The contribution was provided directly to me by some other
person who certified (a), (b) or (c) and I have not modified
it.

**(d)** I understand and agree that this project and the contribution
are public and that a record of the contribution (including all
personal information I submit with it, including my sign-off) is
maintained indefinitely and may be redistributed consistent with
this project or the open source license(s) involved.

---

## How to Sign Off

To sign off your commits, add the following line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

You can automate this with:

```bash
git commit -s -m "Your commit message"
```

Or configure git to always sign off:

```bash
git config --local format.signoff true
```

---

## Why DCO?

The Developer Certificate of Origin ensures:

1. **Provenance**: Clear chain of authorship
2. **Legal clarity**: Contributors affirm they have rights to contribute
3. **Community trust**: Transparent contribution process
4. **License compliance**: Contributions match project licenses

This aligns with RSR (Rhodium Standard Repository) accountability requirements.
