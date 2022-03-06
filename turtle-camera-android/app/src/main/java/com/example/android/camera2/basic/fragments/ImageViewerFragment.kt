/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camera2.basic.fragments

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.util.Base64
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.TextView
import android.widget.Toast
import androidx.core.view.get
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.navArgs
import com.android.volley.DefaultRetryPolicy
import com.android.volley.toolbox.RequestFuture
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import com.example.android.camera.utils.decodeExifOrientation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.nio.charset.Charset
import kotlin.math.max


class ImageViewerFragment : Fragment() {
    //home alone and rich url
    //val URL = "http://192.168.1.169:5000/string"
    //Digitla ocean url
    val URL = "http://46.101.219.222:5000/string"


    /** AndroidX navigation arguments */
    private val args: ImageViewerFragmentArgs by navArgs()

    /** Default Bitmap decoding options */
    private val bitmapOptions = BitmapFactory.Options().apply {
        inJustDecodeBounds = false
        // Keep Bitmaps at less than 1 MP
        if (max(outHeight, outWidth) > DOWNSAMPLE_SIZE) {
            val scaleFactorX = outWidth / DOWNSAMPLE_SIZE + 1
            val scaleFactorY = outHeight / DOWNSAMPLE_SIZE + 1
            inSampleSize = max(scaleFactorX, scaleFactorY)
        }
    }

    /** Bitmap transformation derived from passed arguments */
    private val bitmapTransformation: Matrix by lazy { decodeExifOrientation(args.orientation) }

    /** Flag indicating that there is depth data available for this image */
    private val isDepth: Boolean by lazy { args.depth  }

    /** Data backing our Bitmap viewpager */
    private val bitmapList: MutableList<Bitmap> = mutableListOf()


    private fun createLayout() = RelativeLayout(requireContext()).apply {
        addView(imageViewFactory(),0)
        addView(textView(),1)
    }

    private fun imageViewFactory() = ImageView(requireContext()).apply {
        layoutParams = ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT)
    }

    private fun textView() = TextView(requireContext()).apply {
        layoutParams = ViewGroup.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
    }

    private val  reuest_queue by lazy { Volley.newRequestQueue(requireContext())}

    override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? = createLayout()


    override fun onViewCreated(layout: View, savedInstanceState: Bundle?) {
        super.onViewCreated(layout, savedInstanceState)

        layout as RelativeLayout

        lifecycleScope.launch(Dispatchers.IO) {
            val view = layout.get(0) as ImageView
            val text = layout.get(1) as TextView
            // Load input image file
            val inputBuffer = loadInputBuffer()

            // Load the main JPEG image
            val originalImage = decodeBitmap(inputBuffer, 0, inputBuffer.size)
            activity?.runOnUiThread {
                view.setImageBitmap(originalImage)
                text.setText("Wait Please")
            }
            //TODO show spinner
            //send for validation
            val base64Image = resultToString(inputBuffer)

            val future: RequestFuture<String> = RequestFuture.newFuture()
            val request = object : StringRequest(
                    Method.POST,
                    URL,
                    future,
                    future
            ) {
                override fun getBody(): ByteArray {
                    return base64Image.toByteArray(Charset.defaultCharset())
                }
            }
            request.setRetryPolicy(
                    DefaultRetryPolicy(
                            300000,
                            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
                    )
            )

            reuest_queue.add(request)

            val response = kotlin.runCatching { future.get() }
            checkNotNull(response)
            response.onSuccess {
                val json = JSONObject(response.getOrThrow())
                val base64ValidatedImage = json.getString("marked_image")
                val decodedString = Base64.decode(base64ValidatedImage, Base64.DEFAULT)
                val marked = decodeBitmap(decodedString, 0, decodedString.size)
                val messages = json.getJSONArray("messages")
                var textMessage:String = ""
                for (i in 0 until messages.length()) {
                    val item = messages.getString(i)
                    textMessage+=item
                    textMessage+="\n"
                }
                val valid = json.getBoolean("valid")
                activity?.runOnUiThread {
                    view.setImageBitmap(marked)
                    if(valid) {
                        text.setBackgroundColor(Color.parseColor("#90EE90"))
                        text.setText("Success!")
                    }else{
                        text.setBackgroundColor(Color.parseColor("#FFCCCB"))
                        text.setText(textMessage)
                    }
                }


            }.onFailure {
                activity?.runOnUiThread {
                    text.setText(it.message)
                }
            }
        }
    }

    /** Utility function used to read input file into a byte array */
    private fun loadInputBuffer(): ByteArray {
        val inputFile = File(args.filePath)
        return BufferedInputStream(inputFile.inputStream()).let { stream ->
            ByteArray(stream.available()).also {
                stream.read(it)
                stream.close()
            }
        }
    }


    /** Utility function used to decode a [Bitmap] from a byte array */
    private fun decodeBitmap(buffer: ByteArray, start: Int, length: Int): Bitmap {

        // Load bitmap from given buffer
        val bitmap = BitmapFactory.decodeByteArray(buffer, start, length, bitmapOptions)

        // Transform bitmap orientation using provided metadata
        return Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, bitmapTransformation, true)
    }

    companion object {
        private val TAG = ImageViewerFragment::class.java.simpleName

        /** Maximum size of [Bitmap] decoded */
        private const val DOWNSAMPLE_SIZE: Int = 1024  // 1MP

        /** These are the magic numbers used to separate the different JPG data chunks */
        private val JPEG_DELIMITER_BYTES = arrayOf(-1, -39)

        /**
         * Utility function used to find the markers indicating separation between JPEG data chunks
         */
        private fun findNextJpegEndMarker(jpegBuffer: ByteArray, start: Int): Int {

            // Sanitize input arguments
            assert(start >= 0) { "Invalid start marker: $start" }
            assert(jpegBuffer.size > start) {
                "Buffer size (${jpegBuffer.size}) smaller than start marker ($start)" }

            // Perform a linear search until the delimiter is found
            for (i in start until jpegBuffer.size - 1) {
                if (jpegBuffer[i].toInt() == JPEG_DELIMITER_BYTES[0] &&
                        jpegBuffer[i + 1].toInt() == JPEG_DELIMITER_BYTES[1]) {
                    return i + 2
                }
            }

            // If we reach this, it means that no marker was found
            throw RuntimeException("Separator marker not found in buffer (${jpegBuffer.size})")
        }

        private fun resultToString(result: ByteArray): String {
            val encodedString: String = Base64.encodeToString(result, Base64.DEFAULT)
            return encodedString
        }
    }
}
