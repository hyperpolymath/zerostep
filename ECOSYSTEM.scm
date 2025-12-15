;; SPDX-License-Identifier: AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
;; ECOSYSTEM.scm — zerostep

(ecosystem
  (version "1.0.0")
  (name "zerostep")
  (type "project")
  (purpose "// SPDX-FileCopyrightText: 2024 Joshua Jewell")

  (position-in-ecosystem
    "Part of hyperpolymath ecosystem. Follows RSR guidelines.")

  (related-projects
    (project (name "rhodium-standard-repositories")
             (url "https://github.com/hyperpolymath/rhodium-standard-repositories")
             (relationship "standard")))

  (what-this-is "// SPDX-FileCopyrightText: 2024 Joshua Jewell")
  (what-this-is-not "- NOT exempt from RSR compliance"))
