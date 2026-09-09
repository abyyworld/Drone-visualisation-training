package androidx.annotation;

/**
 * Enough of the annotation for Tracker to compile off-device.
 *
 * It carries no behaviour at all - it tells a static analyser that a parameter is never
 * null - so an empty declaration is the whole of it. It exists here only because the class
 * that uses it is compared against its JavaScript twin by cross_check.sh, and that has to
 * happen outside an Android build.
 */
public @interface NonNull {
}
