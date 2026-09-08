package world.abyy.droneinspection;

import androidx.annotation.Nullable;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * What was found during a flight, written to a file beside the video.
 *
 * WHY A FILE AND NOT JUST THE VIDEO
 *     The recording shows the boxes, which is the evidence, but nobody is going to scrub a
 *     twenty minute flight to find out how many people went past. The number wanted
 *     afterwards is a number, and it needs to survive the app being closed.
 *
 *     A CSV because the tablet has no spreadsheet on it and a phone or laptop opens one
 *     without being asked to install anything. It sits next to the .mp4 with the same name,
 *     so a flight is one video and one table rather than two things to pair up later.
 *
 * WHAT IS IN IT
 *     One row per detection cycle: how long into the flight, how many were in view, and how
 *     many distinct people had been seen by then. Those last two are different questions
 *     and the file keeps them apart, because a running total that goes down is nonsense and
 *     an in view count that only goes up is a lie.
 *
 * IT NEVER STOPS A FLIGHT
 *     Every failure here is swallowed after the first one is recorded. A full disc or a
 *     revoked directory must not interrupt a recording that is going well: losing the
 *     numbers is a nuisance, losing the footage is the flight.
 */
public final class FlightLog {

    private final File file;
    @Nullable
    private BufferedWriter writer;
    @Nullable
    private String failure;
    private final long startedAt;
    private int rows;

    /** Highest total written, so a still can be saved as the count passes each landmark. */
    private int highestSeen;

    public FlightLog(File file) {
        this.file = file;
        this.startedAt = System.currentTimeMillis();
        try {
            writer = new BufferedWriter(new FileWriter(file));
            writer.write("# drone inspection flight log\n");
            writer.write("# started " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.UK)
                    .format(new Date(startedAt)) + "\n");
            writer.write("# in_view is who was on screen at that moment; people_seen is how\n");
            writer.write("# many different people had been seen by then and never goes down.\n");
            writer.write("seconds,in_view,people_seen\n");
        } catch (IOException | RuntimeException problem) {
            failure = String.valueOf(problem.getMessage());
            writer = null;
        }
    }

    public File file() {
        return file;
    }

    @Nullable
    public String failure() {
        return failure;
    }

    /**
     * One detection cycle's worth.
     *
     * Called from the thread that ran the detection, which is where this belongs: it is file
     * I/O, and the main thread is the one drawing the video.
     */
    public synchronized void record(int inView, int peopleSeen) {
        highestSeen = Math.max(highestSeen, peopleSeen);
        if (writer == null) {
            return;
        }
        try {
            writer.write(String.format(Locale.UK, "%.1f,%d,%d%n",
                    (System.currentTimeMillis() - startedAt) / 1000f, inView, peopleSeen));
            rows++;
            // Flushed every so often rather than every row. A flight that ends in a flat
            // battery still leaves almost all of its numbers behind, and a write every few
            // seconds costs nothing.
            if (rows % 25 == 0) {
                writer.flush();
            }
        } catch (IOException | RuntimeException problem) {
            failure = String.valueOf(problem.getMessage());
            close();
        }
    }

    /** The total at the end, for the summary line. */
    public synchronized int highestSeen() {
        return highestSeen;
    }

    public synchronized void close() {
        if (writer == null) {
            return;
        }
        try {
            writer.write(String.format(Locale.UK,
                    "# %d different people over %.0f seconds%n",
                    highestSeen, (System.currentTimeMillis() - startedAt) / 1000f));
            writer.flush();
            writer.close();
        } catch (IOException | RuntimeException ignored) {
            // Nothing useful is left to do about a file that will not close.
        }
        writer = null;
    }
}
