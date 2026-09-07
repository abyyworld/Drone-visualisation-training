package world.abyy.droneinspection;

import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

/**
 * The first screen: pick a mode.
 *
 * The two halves of this app want opposite things from the hardware. Camera is a live
 * native video surface talking RTSP to the drone link. Analyse is a web page working
 * through files that already exist. Putting them behind one door and asking the operator
 * which they want is simpler than a single screen that tries to be both, and it means the
 * live view is never a step someone has to dismiss to look at yesterday's photographs.
 */
public class MenuActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

        findViewById(R.id.open_camera).setOnClickListener(v ->
                startActivity(new Intent(this, LiveActivity.class)));
        findViewById(R.id.open_analyse).setOnClickListener(v ->
                startActivity(new Intent(this, MainActivity.class)));
    }
}
