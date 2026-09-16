#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/* Keep the kernel ABI in C. Rust owns the camera and copies a dequeued frame
 * before returning its buffer. Only single-plane NV12 in the mplane API is
 * supported; an unexpected negotiated format fails before streaming. */
struct cap_camera {
    int fd, streaming;
    unsigned count;
    void *buffers[8];
    size_t lengths[8];
    struct v4l2_format original;
};

void cap_camera_close(struct cap_camera *camera) {
    if (!camera) return;
    if (camera->fd >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (camera->streaming) ioctl(camera->fd, VIDIOC_STREAMOFF, &type);
        for (unsigned i = 0; i < camera->count; ++i)
            if (camera->buffers[i]) munmap(camera->buffers[i], camera->lengths[i]);
        struct v4l2_requestbuffers request = {
            .type = type, .memory = V4L2_MEMORY_MMAP, .count = 0
        };
        ioctl(camera->fd, VIDIOC_REQBUFS, &request);
        if (camera->original.fmt.pix_mp.width)
            ioctl(camera->fd, VIDIOC_S_FMT, &camera->original);
        close(camera->fd);
    }
    free(camera);
}

struct cap_camera *cap_camera_open(const char *path) {
    struct cap_camera *camera = calloc(1, sizeof(*camera));
    if (!camera) return NULL;
    camera->fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (camera->fd < 0) goto fail;
    camera->original.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(camera->fd, VIDIOC_G_FMT, &camera->original)) goto fail;
    struct v4l2_format format = camera->original;
    format.fmt.pix_mp.width = 1920;
    format.fmt.pix_mp.height = 1080;
    format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    format.fmt.pix_mp.field = V4L2_FIELD_NONE;
    if (ioctl(camera->fd, VIDIOC_S_FMT, &format)) goto fail;
    if (format.fmt.pix_mp.width != 1920 || format.fmt.pix_mp.height != 1080 ||
        format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
        format.fmt.pix_mp.num_planes != 1 ||
        format.fmt.pix_mp.plane_fmt[0].bytesperline != 1920) {
        errno = ENOTSUP; goto fail;
    }
    struct v4l2_requestbuffers request = {
        .type = format.type, .memory = V4L2_MEMORY_MMAP, .count = 8
    };
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &request)) goto fail;
    if (!request.count || request.count > 8) { errno = EOVERFLOW; goto fail; }
    camera->count = request.count;
    for (unsigned i = 0; i < camera->count; ++i) {
        struct v4l2_plane plane = {0};
        struct v4l2_buffer buffer = {
            .type = format.type, .memory = V4L2_MEMORY_MMAP,
            .index = i, .length = 1, .m.planes = &plane
        };
        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &buffer)) goto fail;
        void *address = mmap(NULL, plane.length, PROT_READ | PROT_WRITE,
                             MAP_SHARED, camera->fd, plane.m.mem_offset);
        if (address == MAP_FAILED) goto fail;
        camera->buffers[i] = address;
        camera->lengths[i] = plane.length;
        if (ioctl(camera->fd, VIDIOC_QBUF, &buffer)) goto fail;
    }
    if (ioctl(camera->fd, VIDIOC_STREAMON, &format.type)) goto fail;
    camera->streaming = 1;
    return camera;
fail: {
    int saved = errno;
    cap_camera_close(camera);
    errno = saved;
    return NULL;
}}

/* Returns 1 for a complete frame, 0 for timeout, -1 for a fault. */
int cap_camera_next(struct cap_camera *camera, unsigned char *destination,
                    size_t capacity, int64_t *timestamp_ns, uint32_t *sequence,
                    uint32_t *flags, int timeout_ms) {
    struct pollfd poll_fd = { .fd = camera->fd, .events = POLLIN };
    int ready = poll(&poll_fd, 1, timeout_ms);
    if (ready <= 0) return ready;
    struct v4l2_plane plane = {0};
    struct v4l2_buffer buffer = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, .memory = V4L2_MEMORY_MMAP,
        .length = 1, .m.planes = &plane
    };
    if (ioctl(camera->fd, VIDIOC_DQBUF, &buffer)) return -1;
    int result = 1;
    if (buffer.index >= camera->count || plane.data_offset > plane.bytesused ||
        plane.bytesused > camera->lengths[buffer.index] ||
        plane.bytesused - plane.data_offset != 1920 * 1080 * 3 / 2 ||
        capacity < 1920 * 1080 * 3 / 2 || (buffer.flags & V4L2_BUF_FLAG_ERROR)) {
        result = -1;
    } else {
        memcpy(destination, (char *)camera->buffers[buffer.index] + plane.data_offset,
               1920 * 1080 * 3 / 2);
        *timestamp_ns = (int64_t)buffer.timestamp.tv_sec * 1000000000 +
                        buffer.timestamp.tv_usec * 1000;
        *sequence = buffer.sequence;
        *flags = buffer.flags;
    }
    if (ioctl(camera->fd, VIDIOC_QBUF, &buffer)) return -1;
    if (result < 0) errno = EIO;
    return result;
}

#ifdef CAP_PROBE
#include <stdio.h>
#include <time.h>
int main(void) {
    const char *paths[] = {"/dev/video75", "/dev/video111", "/dev/video84",
                          "/dev/video66", "/dev/video102", "/dev/video93"};
    struct cap_camera *cameras[6] = {0};
    unsigned char *bytes = malloc(1920 * 1080 * 3 / 2);
    int trigger = -1, result = 1;
    setbuf(stdout, NULL);
    if (!bytes) goto cleanup;
    for (int i = 0; i < 6; ++i) {
        cameras[i] = cap_camera_open(paths[i]);
        if (!cameras[i]) { perror(paths[i]); goto cleanup; }
        printf("opened %s\n", paths[i]);
    }
    trigger = open("/dev/frame_trigger", O_RDWR | O_NONBLOCK | O_CLOEXEC);
    unsigned fps = 30;
    if (trigger < 0 || ioctl(trigger, 0x40047402, &fps) || ioctl(trigger, 0x7400)) {
        perror("start trigger"); goto cleanup;
    }
    for (int frame = 0; frame < 300; ++frame) {
        for (int camera = 0; camera < 6; ++camera) {
            int64_t timestamp = 0;
            uint32_t sequence = 0, flags = 0;
            int got = cap_camera_next(cameras[camera], bytes, 1920 * 1080 * 3 / 2,
                                      &timestamp, &sequence, &flags, 2000);
            if (got != 1) { fprintf(stderr, "camera %d frame %d got %d: %s\n", camera, frame, got, strerror(errno)); goto cleanup; }
            if (frame < 3 || frame == 299)
                printf("camera=%d frame=%d seq=%u timestamp=%ld flags=%x luma=%u\n",
                       camera, frame, sequence, (long)timestamp, flags, bytes[500000]);
        }
        uint64_t events[2];
        struct pollfd event_fd = { .fd = trigger, .events = POLLIN };
        while (poll(&event_fd, 1, 0) > 0 && (event_fd.revents & POLLIN) &&
               read(trigger, events, sizeof(events)) == sizeof(events)) {
            if (frame < 3) printf("trigger seq=%lu time=%lu\n", (unsigned long)events[0], (unsigned long)events[1]);
        }
    }
    result = 0;
cleanup:
    if (trigger >= 0) { ioctl(trigger, 0x7401); close(trigger); }
    for (int i = 0; i < 6; ++i) cap_camera_close(cameras[i]);
    free(bytes);
    return result;
}
#endif
