package world.abyy.droneinspection;

import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.util.Arrays;
import java.util.List;

/**
 * Where the live screen gets its stream address, its subject and its API key.
 *
 * The stream, the subject, the provider, the model and the interval are saved. The key is
 * not: it lives in memory until the app is closed, for the reason set out in Settings.
 */
public class LiveSettingsActivity extends AppCompatActivity {

    private static final List<String> DOMAINS = Arrays.asList("crowd", "wildfire", "turbine", "solar", "auto");
    private static final List<String> PROVIDERS = Arrays.asList("anthropic", "gemini", "openai");

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_settings);

        EditText stream = findViewById(R.id.stream);
        EditText model = findViewById(R.id.model);
        EditText key = findViewById(R.id.api_key);
        EditText interval = findViewById(R.id.interval);
        Spinner domain = findViewById(R.id.domain);
        Spinner provider = findViewById(R.id.provider);
        Button save = findViewById(R.id.save);
        SeekBar sensitivity = findViewById(R.id.sensitivity);
        TextView sensitivityReadout = findViewById(R.id.sensitivity_readout);

        domain.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, DOMAINS));
        provider.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, PROVIDERS));

        // The bar counts from the floor rather than from zero, because below that floor the
        // decode after the model starts to cost real time and nothing here is allowed to
        // make the tablet slower.
        sensitivity.setMax(Settings.MAX_SENSITIVITY - Settings.MIN_SENSITIVITY);
        sensitivity.setProgress(Settings.sensitivityPercent(this) - Settings.MIN_SENSITIVITY);
        sensitivityReadout.setText(getString(R.string.sensitivity_readout,
                Settings.sensitivityPercent(this)));
        sensitivity.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar bar, int progress, boolean fromUser) {
                sensitivityReadout.setText(getString(R.string.sensitivity_readout,
                        progress + Settings.MIN_SENSITIVITY));
            }

            @Override
            public void onStartTrackingTouch(SeekBar bar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar bar) {
            }
        });

        stream.setText(Settings.stream(this));
        model.setText(Settings.model(this));
        key.setText(Settings.apiKey());
        interval.setText(String.valueOf(Settings.intervalSeconds(this)));
        domain.setSelection(Math.max(0, DOMAINS.indexOf(Settings.domain(this))));
        provider.setSelection(Math.max(0, PROVIDERS.indexOf(Settings.provider(this))));

        save.setOnClickListener(v -> {
            String uri = stream.getText().toString().trim();
            if (!uri.startsWith("rtsp://") && !uri.startsWith("rtsps://")) {
                // Caught here rather than as a playback error thirty seconds later on a
                // screen showing nothing, which is the same symptom as the drone being off.
                Toast.makeText(this, R.string.stream_must_be_rtsp, Toast.LENGTH_LONG).show();
                return;
            }
            int seconds;
            try {
                seconds = Integer.parseInt(interval.getText().toString().trim());
            } catch (NumberFormatException notANumber) {
                seconds = Settings.DEFAULT_INTERVAL_SECONDS;
            }
            Settings.save(this, uri, (String) domain.getSelectedItem(),
                    (String) provider.getSelectedItem(), model.getText().toString(), seconds,
                    sensitivity.getProgress() + Settings.MIN_SENSITIVITY);
            Settings.setApiKey(key.getText().toString());
            finish();
        });
    }
}
