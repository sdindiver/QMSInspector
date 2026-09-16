"""Bracket-specific inspection modules.

- ``holes``          - locate + crop the big splined (serration) hole.
- ``serration``      - present/missing classifier consumed by the recognizer.
- ``train_serration``      - (re)train the serration classifier on big-hole crops.
- ``train_big_hole_yolo``  - (re)train the YOLO big-hole detector used for cropping.

The generic engine in ``inspector/`` calls into ``serration.predict`` for
Bracket parts; everything else here is offline training tooling.
"""
