package com.bruxecho.recorder.playback

import android.content.Context
import android.media.MediaPlayer
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext

private const val ASSET_PREFIX = "asset:///"

/**
 * Plays a single grind-event clip at a time — starting a new one stops
 * whatever was already playing. Understands two path shapes: `asset:///…`
 * for bundled demo clips (see DemoModeState), and a plain filesystem path
 * for anything else, which is what a real clip-cutting pipeline (Sprint 6)
 * will hand it — this class doesn't need to change when that lands.
 */
class ClipPlayer(private val context: Context) {
    private var mediaPlayer: MediaPlayer? = null

    var currentlyPlayingPath by mutableStateOf<String?>(null)
        private set

    fun play(path: String) {
        stop()
        val player = MediaPlayer()
        try {
            if (path.startsWith(ASSET_PREFIX)) {
                val assetPath = path.removePrefix(ASSET_PREFIX)
                context.assets.openFd(assetPath).use { afd ->
                    player.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                }
            } else {
                player.setDataSource(path)
            }
            player.setOnCompletionListener {
                currentlyPlayingPath = null
                it.release()
                if (mediaPlayer === it) mediaPlayer = null
            }
            player.prepare()
            player.start()
            mediaPlayer = player
            currentlyPlayingPath = path
        } catch (e: Exception) {
            player.release()
            currentlyPlayingPath = null
        }
    }

    fun stop() {
        mediaPlayer?.let { runCatching { it.stop() }; it.release() }
        mediaPlayer = null
        currentlyPlayingPath = null
    }
}

@Composable
fun rememberClipPlayer(): ClipPlayer {
    val context = LocalContext.current
    val player = remember { ClipPlayer(context) }
    DisposableEffect(Unit) { onDispose { player.stop() } }
    return player
}
