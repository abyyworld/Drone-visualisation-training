#!/usr/bin/env bash
#
# The flame and smoke scanner exists twice: web/js/firescan.js runs in the browser and in
# the APK's web view, android/.../FireScan.java runs on the drone's RTSP feed. Two
# implementations of one method is a standing invitation to drift, and drift here means the
# app marks a region on the tablet that the report of the same footage does not.
#
# So both are run over the same eight painted frames and the output is compared exactly:
# same labels, same confidences to two places, same boxes. A change to one that is not
# mirrored in the other fails here rather than in the field.
#
#   bash tests/java/cross_check.sh
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

# Tiles, Yolo and the Tracker are compiled alongside FireScan. All of them exist twice,
# once in Java for
# the drone's feed and once in JavaScript for the browser, and all of them are compared
# below. Yolo is the box decode, where a mistake does not raise an error: it draws boxes
# beside people, or finds nobody at all and looks exactly like an empty frame. The
# tracker is where the numbers on screen come from, and two of them that disagree means
# the tablet and the report of the same footage count the same crowd differently.
javac -nowarn -d "$out" \
  "$here/android/graphics/Bitmap.java" \
  "$here/stub/Finding.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/FireScan.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/Tiles.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/Yolo.java" \
  "$here/androidx/annotation/NonNull.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/Reid.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/Tracker.java" \
  "$root/android/app/src/main/java/world/abyy/droneinspection/Letterbox.java" \
  "$here/Cross.java"

java -cp "$out" Cross > "$out/java.txt"
node "$here/cross.mjs" > "$out/js.txt"

if diff -u "$out/java.txt" "$out/js.txt"; then
  echo
  echo "Java and JavaScript agree on all $(wc -l < "$out/js.txt") cases."
else
  echo
  echo "The Java port and the JavaScript disagree. One of them has been changed alone."
  exit 1
fi
