package com.bruxecho.recorder

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.bruxecho.recorder.domain.DemoModeState

const val RECORDING_NOTIFICATION_CHANNEL_ID = "bruxecho_recording"

class BruxEchoApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        DemoModeState.init(this)
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
