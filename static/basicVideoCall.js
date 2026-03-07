/*
 *  These procedures use Agora Video Call SDK for Web to enable local and remote
 *  users to join and leave a Video Call channel managed by Agora Platform.
 */

/*
 *  Create an {@link https://docs.agora.io/en/Video/API%20Reference/web_ng/interfaces/iagorartcclient.html|AgoraRTCClient} instance.
 *
 * @param {string} mode - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/clientconfig.html#mode| streaming algorithm} used by Agora SDK.
 * @param  {string} codec - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/clientconfig.html#codec| client codec} used by the browser.
 */
var client;

/*
 * Clear the video and audio tracks used by `client` on initiation.
 */
var localTracks = {
  videoTrack: null,
  audioTrack: null,
};

/*
 * On initiation no users are connected.
 */
var remoteUsers = {};

const DEBUG_MODE = false;
const lastBase64Frames = {};
const subscribeState = new Map();
const videoSlots = {
  frontUid: null,
  rearUid: null,
};
const defaultVideoStatus = {
  joinedRtc: false,
  remoteVideoPublished: false,
  remoteVideoUids: [],
  frontUid: null,
  rearUid: null,
  severity: "secondary",
  message: "RTC not joined.",
  lastError: null,
};
window.sdkVideoStatus = { ...defaultVideoStatus };
window.imageParams = {
  imageFormat: "png",
  imageQuality: 1.0,
};

function renderVideoStatus() {
  const el = document.getElementById("video-status-alert");
  if (!el) {
    return;
  }
  const status = window.sdkVideoStatus || defaultVideoStatus;
  el.className = `alert alert-${status.severity || "secondary"} fade show`;
  el.textContent = status.message || "";
  el.style.display = status.message ? "block" : "none";
}

function getErrorText(error) {
  if (!error) {
    return "unknown error";
  }
  if (typeof error === "string") {
    return error;
  }
  if (error.message) {
    return error.message;
  }
  return String(error);
}

function isRepeatSubscribeError(error) {
  const text = getErrorText(error);
  return (
    text.includes("Repeat subscribe request") ||
    text.includes("ERR_SUBSCRIBE_REQUEST_INVALID")
  );
}

function getSubscribeKey(uid, mediaType) {
  return `${Number(uid)}:${String(mediaType)}`;
}

function clearSubscribeStateForUid(uid) {
  const normalizedUid = Number(uid);
  for (const key of subscribeState.keys()) {
    if (key.startsWith(`${normalizedUid}:`)) {
      subscribeState.delete(key);
    }
  }
}

function setVideoStatus(patch) {
  window.sdkVideoStatus = {
    ...window.sdkVideoStatus,
    ...patch,
  };
  renderVideoStatus();
}

function getPublishedVideoUids() {
  return Object.values(remoteUsers)
    .filter((user) => user && user.videoTrack)
    .map((user) => Number(user.uid))
    .filter((uid) => Number.isFinite(uid));
}

function assignVideoSlot(uid) {
  const normalizedUid = Number(uid);
  if (!Number.isFinite(normalizedUid)) {
    return;
  }
  if (normalizedUid === 1000) {
    videoSlots.frontUid = normalizedUid;
    return;
  }
  if (normalizedUid === 1001) {
    videoSlots.rearUid = normalizedUid;
    return;
  }
  if (videoSlots.frontUid === null) {
    videoSlots.frontUid = normalizedUid;
    return;
  }
  if (videoSlots.frontUid !== normalizedUid && videoSlots.rearUid === null) {
    videoSlots.rearUid = normalizedUid;
  }
}

function rebuildVideoSlots() {
  const publishedUids = getPublishedVideoUids();
  videoSlots.frontUid = publishedUids.includes(1000) ? 1000 : null;
  videoSlots.rearUid = publishedUids.includes(1001) ? 1001 : null;
  publishedUids.forEach((uid) => assignVideoSlot(uid));

  const remoteVideoPublished = publishedUids.length > 0;
  const message = remoteVideoPublished
    ? `Receiving remote video from UID${publishedUids.length > 1 ? "s" : ""} ${publishedUids.join(", ")}.`
    : window.sdkVideoStatus.joinedRtc
      ? "Joined RTC channel. Waiting for remote video to be published."
      : "RTC not joined.";

  setVideoStatus({
    remoteVideoPublished,
    remoteVideoUids: publishedUids,
    frontUid: videoSlots.frontUid,
    rearUid: videoSlots.rearUid,
    severity: remoteVideoPublished
      ? "success"
      : window.sdkVideoStatus.joinedRtc
        ? "warning"
        : "secondary",
    message,
  });
}

function scheduleRemoteVideoWarning() {
  if (window.remoteVideoWaitTimer) {
    clearTimeout(window.remoteVideoWaitTimer);
  }
  window.remoteVideoWaitTimer = window.setTimeout(() => {
    if (window.sdkVideoStatus.joinedRtc && getPublishedVideoUids().length === 0) {
      setVideoStatus({
        remoteVideoPublished: false,
        remoteVideoUids: [],
        severity: "warning",
        message: "Joined RTC channel, but no remote video has been published yet.",
      });
    }
  }, 10000);
}

/*
 * On initiation. `client` is not attached to any project or channel for any specific user.
 */
var options = {
  appid: null,
  channel: null,
  uid: null,
  token: null,
};

// you can find all the agora preset video profiles here https://docs.agora.io/en/Voice/API%20Reference/web_ng/globals.html#videoencoderconfigurationpreset
var videoProfiles = [
  {
    label: "360p_7",
    detail: "480×360, 15fps, 320Kbps",
    value: "360p_7",
  },
  {
    label: "360p_8",
    detail: "480×360, 30fps, 490Kbps",
    value: "360p_8",
  },
  {
    label: "480p_1",
    detail: "640×480, 15fps, 500Kbps",
    value: "480p_1",
  },
  {
    label: "480p_2",
    detail: "640×480, 30fps, 1000Kbps",
    value: "480p_2",
  },
  {
    label: "720p_1",
    detail: "1280×720, 15fps, 1130Kbps",
    value: "720p_1",
  },
  {
    label: "720p_2",
    detail: "1280×720, 30fps, 2000Kbps",
    value: "720p_2",
  },
  {
    label: "1080p_1",
    detail: "1920×1080, 15fps, 2080Kbps",
    value: "1080p_1",
  },
  {
    label: "1080p_2",
    detail: "1920×1080, 30fps, 3000Kbps",
    value: "1080p_2",
  },
];
var curVideoProfile;
AgoraRTC.onAutoplayFailed = () => {
  alert("click to start autoplay!");
};
AgoraRTC.onMicrophoneChanged = async (changedDevice) => {
  // When plugging in a device, switch to a device that is newly plugged in.
  if (changedDevice.state === "ACTIVE") {
    localTracks.audioTrack.setDevice(changedDevice.device.deviceId);
    // Switch to an existing device when the current device is unplugged.
  } else if (
    changedDevice.device.label === localTracks.audioTrack.getTrackLabel()
  ) {
    const oldMicrophones = await AgoraRTC.getMicrophones();
    oldMicrophones[0] &&
      localTracks.audioTrack.setDevice(oldMicrophones[0].deviceId);
  }
};
AgoraRTC.onCameraChanged = async (changedDevice) => {
  // When plugging in a device, switch to a device that is newly plugged in.
  if (changedDevice.state === "ACTIVE") {
    localTracks.videoTrack.setDevice(changedDevice.device.deviceId);
    // Switch to an existing device when the current device is unplugged.
  } else if (
    changedDevice.device.label === localTracks.videoTrack.getTrackLabel()
  ) {
    const oldCameras = await AgoraRTC.getCameras();
    oldCameras[0] && localTracks.videoTrack.setDevice(oldCameras[0].deviceId);
  }
};
async function initDevices() {
  mics = await AgoraRTC.getMicrophones();
  const audioTrackLabel = localTracks.audioTrack.getTrackLabel();
  currentMic = mics.find((item) => item.label === audioTrackLabel);
  $(".mic-input").val(currentMic.label);
  $(".mic-list").empty();
  mics.forEach((mic) => {
    $(".mic-list").append(`<a class="dropdown-item" href="#">${mic.label}</a>`);
  });

  cams = await AgoraRTC.getCameras();
  const videoTrackLabel = localTracks.videoTrack.getTrackLabel();
  currentCam = cams.find((item) => item.label === videoTrackLabel);
  $(".cam-input").val(currentCam.label);
  $(".cam-list").empty();
  cams.forEach((cam) => {
    $(".cam-list").append(`<a class="dropdown-item" href="#">${cam.label}</a>`);
  });
}
async function switchCamera(label) {
  currentCam = cams.find((cam) => cam.label === label);
  $(".cam-input").val(currentCam.label);
  // switch device of local video track.
  await localTracks.videoTrack.setDevice(currentCam.deviceId);
}
async function switchMicrophone(label) {
  currentMic = mics.find((mic) => mic.label === label);
  $(".mic-input").val(currentMic.label);
  // switch device of local audio track.
  await localTracks.audioTrack.setDevice(currentMic.deviceId);
}
function initVideoProfiles() {
  videoProfiles.forEach((profile) => {
    $(".profile-list").append(
      `<a class="dropdown-item" label="${profile.label}" href="#">${profile.label}: ${profile.detail}</a>`
    );
  });
  curVideoProfile = videoProfiles.find((item) => item.label == "480p_1");
  $(".profile-input").val(`${curVideoProfile.detail}`);
}
async function changeVideoProfile(label) {
  curVideoProfile = videoProfiles.find((profile) => profile.label === label);
  $(".profile-input").val(`${curVideoProfile.detail}`);
  // change the local video track`s encoder configuration
  localTracks.videoTrack &&
    (await localTracks.videoTrack.setEncoderConfiguration(
      curVideoProfile.value
    ));
}

/*
 * When this page is called with parameters in the URL, this procedure
 * attempts to join a Video Call channel using those parameters.
 */
$(() => {
  initVideoProfiles();
  renderVideoStatus();
  $(".profile-list").delegate("a", "click", function (e) {
    changeVideoProfile(this.getAttribute("label"));
  });
  var urlParams = new URL(location.href).searchParams;
  options.appid = urlParams.get("appid");
  options.channel = urlParams.get("channel");
  options.token = urlParams.get("token");
  options.uid = urlParams.get("uid");
  if (options.appid && options.channel) {
    $("#uid").val(options.uid);
    $("#appid").val(options.appid);
    $("#token").val(options.token);
    $("#channel").val(options.channel);
    $("#join-form").submit();
  }
});

/*
 * When a user clicks Join or Leave in the HTML form, this procedure gathers the information
 * entered in the form and calls join asynchronously. The UI is updated to match the options entered
 * by the user.
 */
$("#join-form").submit(async function (e) {
  e.preventDefault();
  $("#join").attr("disabled", true);
  try {
    client = AgoraRTC.createClient({
      mode: "rtc",
      codec: getCodec(),
    });
    options.channel = $("#channel").val();
    options.uid = Number($("#uid").val());
    options.appid = $("#appid").val();
    options.token = $("#token").val();
    remoteUsers = {};
    videoSlots.frontUid = null;
    videoSlots.rearUid = null;
    setVideoStatus({
      ...defaultVideoStatus,
      severity: "info",
      message: "Joining RTC channel...",
    });
    await join();
    if (options.token) {
      $("#success-alert-with-token").css("display", "block");
    } else {
      $("#success-alert a").attr(
        "href",
        `index.html?appid=${options.appid}&channel=${options.channel}&token=${options.token}`
      );
      $("#success-alert").css("display", "block");
    }
  } catch (error) {
    console.error(error);
    setVideoStatus({
      joinedRtc: false,
      remoteVideoPublished: false,
      severity: "danger",
      message: `RTC join failed: ${error}`,
      lastError: String(error),
    });
  } finally {
    $("#leave").attr("disabled", false);
  }
});

/*
 * Called when a user clicks Leave in order to exit a channel.
 */
$("#leave").click(function (e) {
  leave();
});
$("#agora-collapse").on("show.bs.collapse	", function () {
  initDevices();
});
$(".cam-list").delegate("a", "click", function (e) {
  switchCamera(this.text);
});
$(".mic-list").delegate("a", "click", function (e) {
  switchMicrophone(this.text);
});

/*
 * Join a channel, then create local video and audio tracks and publish them to the channel.
 */
async function join() {
  // Add an event listener to play remote tracks when remote user publishes.
  client.on("user-published", handleUserPublished);
  client.on("user-unpublished", handleUserUnpublished);
  client.on("user-joined", handleUserJoined);
  client.on("user-left", handleUserLeft);
  client.on("user-info-updated", handleUserInfoUpdated);
  options.uid = await client.join(
    options.appid,
    options.channel,
    options.token || null,
    options.uid || null
  );
  setVideoStatus({
    joinedRtc: true,
    severity: "info",
    message: "Joined RTC channel. Waiting for remote video to be published.",
    lastError: null,
  });
  scheduleRemoteVideoWarning();
  $("#captured-frames").css("display", DEBUG_MODE ? "block" : "none");
}

/*
 * Stop all local and remote tracks then leave the channel.
 */
async function leave() {
  for (trackName in localTracks) {
    var track = localTracks[trackName];
    if (track) {
      track.stop();
      track.close();
      localTracks[trackName] = undefined;
    }
  }

  // Remove remote users and player views.
  remoteUsers = {};
  subscribeState.clear();
  $("#remote-playerlist").html("");

  // leave the channel
  await client.leave();
  if (window.remoteVideoWaitTimer) {
    clearTimeout(window.remoteVideoWaitTimer);
  }
  videoSlots.frontUid = null;
  videoSlots.rearUid = null;
  setVideoStatus({
    ...defaultVideoStatus,
    message: "Left RTC channel.",
  });
  $("#local-player-name").text("");
  $("#join").attr("disabled", false);
  $("#leave").attr("disabled", true);
  $("#joined-setup").css("display", "none");
  console.log("client leaves channel success");
}

/*
 * Add the local use to a remote channel.
 *
 * @param  {IAgoraRTCRemoteUser} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to add.
 * @param {trackMediaType - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/itrack.html#trackmediatype | media type} to add.
 */
async function subscribe(user, mediaType) {
  const uid = user.uid;
  const subscribeKey = getSubscribeKey(uid, mediaType);
  const existingState = subscribeState.get(subscribeKey);

  if (existingState === "pending" || existingState === "ready") {
    console.warn(
      `Skipping duplicate subscribe for user ${uid}, mediaType ${mediaType} (${existingState})`
    );
    return;
  }

  subscribeState.set(subscribeKey, "pending");
  try {
    await client.subscribe(user, mediaType);
    subscribeState.set(subscribeKey, "ready");
    console.log("subscribe success");
  } catch (error) {
    subscribeState.delete(subscribeKey);
    const errorText = getErrorText(error);
    console.error(`subscribe failed for user ${uid}, mediaType ${mediaType}`, error);
    if (mediaType === "video") {
      setVideoStatus({
        joinedRtc: true,
        severity: "danger",
        message: `Remote user ${uid} published video, but browser subscribe failed: ${errorText}`,
        lastError: errorText,
      });
    }
    throw error;
  }

  if (mediaType === "video") {
    if ($(`#player-wrapper-${uid}`).length > 0) {
      $(`#player-wrapper-${uid}`).remove();
    }
    if ($(`#captured-frame-${uid}`).length > 0) {
      $(`#captured-frame-${uid}`).remove();
    }
    const playerWidth =
      uid === 1001 ? "540px" : uid === 1000 ? "1024px" : "auto";
    const playerHeight =
      uid === 1001 ? "360px" : uid === 1000 ? "576px" : "auto";

    const player = $(`
      <div id="player-wrapper-${uid}">
        <p class="player-name">(${uid})</p>
        <div id="player-${uid}" class="player" style="width: ${playerWidth}; height: ${playerHeight};"></div>
      </div>
    `);
    $("#remote-playerlist").append(player);
    user.videoTrack.play(`player-${uid}`);

    const capturedFrameDiv = $(`
      <div id="captured-frame-${uid}" style="width: ${playerWidth}; height: ${playerHeight}; display: ${
      DEBUG_MODE ? "block" : "none"
    };">
        <p>Captured Frames (${uid})</p>
        <img id="captured-image-${uid}" style="width: 100%; height: 100%; object-fit: contain;">
        <button id="download-frame-${uid}" class="btn btn-primary mt-2">Download Frame</button>
        <button id="download-base64-${uid}" class="btn btn-secondary mt-2 ml-2">Download Base64</button>
      </div>
    `);
    $("#captured-frames").append(capturedFrameDiv);

    user.videoTrack.captureEnabled = true;
    assignVideoSlot(uid);
    rebuildVideoSlots();
    if (window.remoteVideoWaitTimer) {
      clearTimeout(window.remoteVideoWaitTimer);
    }
  }
  if (mediaType === "audio") {
    user.audioTrack.play();
  }
}

/*
 * Add a user who has subscribed to the live channel to the local interface.
 *
 * @param  {IAgoraRTCRemoteUser} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to add.
 * @param {trackMediaType - The {@link https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/itrack.html#trackmediatype | media type} to add.
 */
function handleUserPublished(user, mediaType) {
  const id = user.uid;
  remoteUsers[id] = user;
  if (mediaType === "video") {
    setVideoStatus({
      joinedRtc: true,
      severity: "info",
      message: `Remote user ${id} published video. Subscribing...`,
    });
  }
  void subscribe(user, mediaType).catch((error) => {
    if (isRepeatSubscribeError(error)) {
      console.warn(`Ignoring repeat subscribe response for user ${id}, mediaType ${mediaType}`, error);
      return;
    }
  });
}

function handleUserJoined(user) {
  const uid = Number(user.uid);
  if (!Number.isFinite(uid)) {
    return;
  }
  setVideoStatus({
    joinedRtc: true,
    severity: "info",
    message: `Remote user ${uid} joined. Waiting for video publication.`,
  });
}

function handleUserLeft(user) {
  const uid = Number(user.uid);
  delete remoteUsers[uid];
  clearSubscribeStateForUid(uid);
  rebuildVideoSlots();
  if (window.sdkVideoStatus.joinedRtc && !window.sdkVideoStatus.remoteVideoPublished) {
    setVideoStatus({
      severity: "warning",
      message: `Remote user ${uid} left before publishing video.`,
    });
  }
}

function handleUserInfoUpdated(uidOrUser, msg) {
  const uid =
    typeof uidOrUser === "object" && uidOrUser !== null
      ? Number(uidOrUser.uid)
      : Number(uidOrUser);
  if (!Number.isFinite(uid)) {
    return;
  }
  if (msg === "mute-video" && getPublishedVideoUids().length === 0) {
    setVideoStatus({
      joinedRtc: true,
      severity: "warning",
      message: `Remote user ${uid} is connected, but video is muted or not published.`,
    });
  }
}

/*
 * Remove the user specified from the channel in the local interface.
 *
 * @param  {string} user - The {@link  https://docs.agora.io/en/Voice/API%20Reference/web_ng/interfaces/iagorartcremoteuser.html| remote user} to remove.
 */
function handleUserUnpublished(user, mediaType) {
  subscribeState.delete(getSubscribeKey(user.uid, mediaType));
  if (mediaType === "video") {
    const id = user.uid;
    delete remoteUsers[id];
    $(`#player-wrapper-${id}`).remove();
    $(`#captured-frame-${id}`).remove();
    rebuildVideoSlots();
  }
}
function getCodec() {
  var radios = document.getElementsByName("radios");
  var value;
  for (var i = 0; i < radios.length; i++) {
    if (radios[i].checked) {
      value = radios[i].value;
    }
  }
  return value;
}

async function captureFrameAsBase64(videoTrack, uid) {
  // Try to capture from video element directly (works in WebKit/Safari)
  const videoElement = document.querySelector(`#player-${uid} video`);
  
  if (videoElement && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(
      `image/${window.imageParams["imageFormat"]}`,
      window.imageParams["imageQuality"]
    );
  }
  
  // Fallback to Agora's API (works in Chrome when not headless)
  try {
    const frame = await videoTrack.getCurrentFrameData();
    if (frame && frame.width > 0 && frame.height > 0) {
      const canvas = document.createElement("canvas");
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext("2d");
      ctx.putImageData(frame, 0, 0);
      return canvas.toDataURL(
        `image/${window.imageParams["imageFormat"]}`,
        window.imageParams["imageQuality"]
      );
    }
  } catch (e) {
    console.warn("getCurrentFrameData failed:", e);
  }
  
  return null;
}

// Function to get the latest base64 frame for a specific UID
async function getLastBase64Frame(uid) {
  const user = remoteUsers[uid];
  if (!user || !user.videoTrack || !user.videoTrack.captureEnabled) {
    return null;
  }

  const base64Frame = await captureFrameAsBase64(user.videoTrack, uid);
  lastBase64Frames[uid] = base64Frame;
  return base64Frame;
}

function initializeImageParams({ imageFormat, imageQuality }) {
  window.imageParams = { imageFormat, imageQuality };
}

function getRemoteVideoStatus() {
  return {
    ...window.sdkVideoStatus,
    frontUid: videoSlots.frontUid,
    rearUid: videoSlots.rearUid,
    remoteVideoUids: getPublishedVideoUids(),
  };
}

function getPreferredVideoUid(view) {
  const status = getRemoteVideoStatus();
  if (view === "rear") {
    return status.rearUid;
  }
  return status.frontUid;
}

async function getLastBase64FrameForView(view) {
  const uid = getPreferredVideoUid(view);
  if (uid === null || uid === undefined) {
    return null;
  }
  return getLastBase64Frame(uid);
}

window.initializeImageParams = initializeImageParams;
window.getLastBase64Frame = getLastBase64Frame;
window.getLastBase64FrameForView = getLastBase64FrameForView;
window.getRemoteVideoStatus = getRemoteVideoStatus;
