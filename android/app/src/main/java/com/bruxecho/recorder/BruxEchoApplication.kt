package com.bruxecho.recorder

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.bruxecho.recorder.data.SessionSync
import com.bruxecho.recorder.domain.DemoModeState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

const val RECORDING_NOTIFICATION_CHANNEL_ID = "bruxecho_recording"

class BruxEchoApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        DemoModeState.init(this)
        // Fire-and-forget: mirrors any session.json Room doesn't know about yet
        // (see SessionSync doc comment). Repository calls also sync defensively,
        // so a screen opened before this finishes still sees correct data.
        CoroutineScope(Dispatchers.IO).launch { SessionSync.sync(this@BruxEchoApplication) }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                RECORDING_NOTIFICATION_CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
}
