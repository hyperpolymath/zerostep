;; ZeroStep - Guix Package Definition
;; Run: guix shell -D -f guix.scm

(use-modules (guix packages)
             (guix gexp)
             (guix git-download)
             (guix build-system cargo)
             ((guix licenses) #:prefix license:)
             (gnu packages base))

(define-public zerostep
  (package
    (name "ZeroStep")
    (version "0.1.0")
    (source (local-file "." "ZeroStep-checkout"
                        #:recursive? #t
                        #:select? (git-predicate ".")))
    (build-system cargo-build-system)
    (synopsis "Rust application")
    (description "Rust application - part of the RSR ecosystem.")
    (home-page "https://github.com/hyperpolymath/ZeroStep")
    (license license:agpl3+)))

;; Return package for guix shell
zerostep
