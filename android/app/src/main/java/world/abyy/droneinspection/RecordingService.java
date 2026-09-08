package world.abyy.droneinspection;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;

import androidx.annotation.Nullable;

/**
 * Keeps the recording alive while the app is not on screen.
 *
 * WHY THIS IS NEEDED
 *     A pilot does not sit in this app. They switch to the flight software, they look at a
 *     map, the screen times out. Android is entitled to stop an activity that is not
 *     visible and, before long, to kill the process behind it - and a recording that ends
 *     silently when someone changes app is worse than one that never started, because
 *     nobody finds out until they land.
 *
 *     A foreground service is the only thing that tells Android this process is doing
 *     something the person asked for and can see. The price is a notification that cannot
 *     be dismissed, which is the right price: an app quietly recording in the background
 *     with nothing on screen to say so is exactly what that rule exists to prevent.
 *
 * WHAT IT DOES NOT DO
 *     No work happens in this class. The video pipeline, the encoder and the detector all
 *     stay where they are, in the activity, and carry on running because the process is now
 *     allowed to. This exists to hold the process up, and to put a line in the shade that
 *     says a recording is running and taps back into it.
 */
public final class RecordingService extends Service {

    private static final String CHANNEL = "recording";
    private static final int NOTIFICATION = 1;

    static void start(Context context) {
        Intent intent = new Intent(context, RecordingService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(intent);
        } else {
            context.startService(intent);
        }
    }

    static void stop(Context context) {
        context.stopService(new Intent(context, RecordingService.class));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        createChannel();

        Intent open = new Intent(this, LiveActivity.class);
        open.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent tap = PendingIntent.getActivity(
                this, 0, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification notification = new Notification.Builder(this, CHANNEL)
                .setContentTitle(getString(R.string.recording_notification_title))
                .setContentText(getString(R.string.recording_notification_text))
                .setSmallIcon(R.drawable.ic_recording)
                .setOngoing(true)
                .setContentIntent(tap)
                .build();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // A type is mandatory from Android 10 and the choice is checked from 14. This
            // service exists to keep a stream being read and written to a file, which is
            // what dataSync describes; it is not projecting a screen and not using a
            // camera, and claiming either of those would be a false declaration.
            startForeground(NOTIFICATION, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(NOTIFICATION, notification);
        }

        // Not restarted if the system kills the process: by then the encoder and the
        // pipeline are gone with it, and a service that came back alone would hold a
        // notification over a recording that no longer exists.
        return START_NOT_STICKY;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return;
        }
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager == null || manager.getNotificationChannel(CHANNEL) != null) {
            return;
        }
        NotificationChannel channel = new NotificationChannel(
                CHANNEL, getString(R.string.recording_channel), NotificationManager.IMPORTANCE_LOW);
        channel.setDescription(getString(R.string.recording_channel_description));
        // Silent: this appears the moment recording starts and stays for the whole flight.
        // A sound would be an alert about something the operator just did on purpose.
        channel.setSound(null, null);
        channel.enableVibration(false);
        manager.createNotificationChannel(channel);
    }
}
