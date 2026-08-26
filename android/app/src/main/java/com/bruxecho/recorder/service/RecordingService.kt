package com.bruxecho.recorder.service

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.bruxecho.recorder.R
import com.bruxecho.recorder.RECORDING_NOTIFICATION_CHANNEL_ID
import com.bruxecho.recorder.audio.RecordingEngine
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

private const val NOTIFICATION_ID = 1001
private const val REQUESTED_SAMPLE_RATE_HZ = 44_100
private const val NOTIFICATION_UPDATE_INTERVAL_MS = 1_000L

/**
 * Foreground service hosting [RecordingEngine] for the lifetime of an
 * overnight recording. Owns the pieces that must outlive the Activity:
 * the wake lock, the notification, and audio-focus arbitration against
 * phone calls.
 */
class RecordingService : Service() {

    companion object {
        const val ACTION_START = "com.bruxecho.recorder.action.START"
        const val ACTION_STOP = "com.bruxecho.recorder.action.STOP"

        fun start(context: Context) {
            val intent = Intent(context, RecordingService::class.java).setAction(ACTION_START)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, RecordingService::class.java).setAction(ACTION_STOP)
            context.startService(intent)
        }
    }

    private var engine: RecordingEngine? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var audioManager: AudioManager? = null
    private var focusRequest: AudioFocusRequest? = null
    private val notificationHandler = Handler(Looper.getMainLooper())
    private var notificationTicker: Runnable? = null

    private val focusChangeListener = AudioManager.OnAudioFocusChangeListener { change ->
        when (change) {
            AudioManager.AUDIOFOCUS_LOSS,
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT,
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> engine?.pause("audio_focus_lost")
            AudioManager.AUDIOFOCUS_GAIN -> engine?.resume()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopRecording()
                stopSelf()
            }
            else -> startRecording()
        }
        return START_NOT_STICKY
    }

    private fun startRecording() {
        if (engine != null) return

        startForeground(NOTIFICATION_ID, buildNotification(0.0))
        acquireWakeLock()
        requestAudioFocus()

        val sessionId = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss", Locale.US))
        val baseDir = getExternalFilesDir(null) ?: filesDir
        val sessionDir = File(baseDir, "sessions/$sessionId")
        sessionDir.mkdirs()

        val newEngine = RecordingEngine(sessionDir, REQUESTED_SAMPLE_RATE_HZ)
        newEngine.onTick = { elapsedTotalS, rms, bytesTotal ->
            RecordingState.update {
                it.copy(
                    isRecording = true,
                    sessionId = newEngine.sessionId,
                    elapsedTotalS = elapsedTotalS,
                    currentRms = rms,
                    sourceLabel = newEngine.sourceLabel,
                    sampleRateActual = newEngine.sampleRateActual,
                    bytesWrittenTotal = bytesTotal,
                )
            }
        }
        newEngine.onAutoStopLowStorage = {
            RecordingState.update { it.copy(lastError = "พื้นที่เก็บข้อมูลเหลือน้อยเกินไป หยุดบันทึกอัตโนมัติ") }
            stopRecording()
            stopSelf()
        }
        newEngine.onFatalError = { message ->
            RecordingState.update { it.copy(lastError = message) }
            stopRecording()
            stopSelf()
        }

        val deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}"
        val androidVersion = Build.VERSION.RELEASE ?: Build.VERSION.SDK_INT.toString()
        val opened = newEngine.start(deviceModel, androidVersion)
        if (!opened) {
            RecordingState.update { it.copy(lastError = "ไม่สามารถเปิดไมโครโฟนได้") }
            stopSelf()
            return
        }

        engine = newEngine
        startNotificationTicker()
    }

    private fun stopRecording() {
        notificationTicker?.let { notificationHandler.removeCallbacks(it) }
        notificationTicker = null
        engine?.stop()
        engine = null
        releaseAudioFocus()
        releaseWakeLock()
        RecordingState.update { it.copy(isRecording = false) }
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    private fun startNotificationTicker() {
        val ticker = object : Runnable {
            override fun run() {
                val elapsed = RecordingState.state.value.elapsedTotalS
                val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                manager.notify(NOTIFICATION_ID, buildNotification(elapsed))
                notificationHandler.postDelayed(this, NOTIFICATION_UPDATE_INTERVAL_MS)
            }
        }
        notificationTicker = ticker
        notificationHandler.postDelayed(ticker, NOTIFICATION_UPDATE_INTERVAL_MS)
    }

    private fun buildNotification(elapsedTotalS: Double): Notification {
        val stopIntent = Intent(this, RecordingService::class.java).setAction(ACTION_STOP)
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val totalSeconds = elapsedTotalS.toLong()
        val h = totalSeconds / 3600
        val m = (totalSeconds % 3600) / 60
        val s = totalSeconds % 60
        val timeText = "%02d:%02d:%02d".format(h, m, s)

        return NotificationCompat.Builder(this, RECORDING_NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(timeText)
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .addAction(0, getString(R.string.notification_stop_action), stopPendingIntent)
            .build()
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "BruxEcho:RecordingWakeLock").apply {
            setReferenceCounted(false)
            acquire(9 * 60 * 60 * 1000L) // 9h safety cap; released explicitly on stop
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun requestAudioFocus() {
        val manager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager = manager
        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_UNKNOWN)
            .build()
        val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(attributes)
            .setOnAudioFocusChangeListener(focusChangeListener, notificationHandler)
            .build()
        focusRequest = request
        manager.requestAudioFocus(request)
    }

    private fun releaseAudioFocus() {
        focusRequest?.let { audioManager?.abandonAudioFocusRequest(it) }
        focusRequest = null
    }

    override fun onDestroy() {
        stopRecording()
        super.onDestroy()
    }
}
