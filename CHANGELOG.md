# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


BalloonLib is a physics-informed neural network library for modeling 
hemodynamic response functions using the Balloon model. It embeds the Balloon haemodynamic model as a physics constraint inside a multi-headed neural network, enabling data-driven HRF estimation from fMRI BOLD signals while respecting physiological dynamics.

## [Unreleased]

### Added
- 


### Changed
- 

### Deprecated
- 

### Removed
- 

### Fixed
- 

### Security
- 

---

## [0.1.0] - 2026-04-24

### Added
- Initial release of BalloonLib
- Balloon model physiological equations (`balloonmodellib.py`)
- Physics constraints for PINN (`physics.py`)
- Neural network layer implementations (`layers.py`)
- Multihead PINN model architecture (`model.py`)
- Training loop and loss functions (`training.py`)
- Data handling utilities (`data.py`)
- HRF descriptor metrics (HP, TTP, FWHM, etc.) (`metrics.py`)
- Visualization utilities (`plotting.py`)
- Random Weight Factorization layers (`rwf_layers.py`)

---

[Unreleased]: https://github.com/errehache/BalloonLib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/errehache/BalloonLib/releases/tag/v0.1.0