/* V4L2 multi-planar NV12 capture with monotonic timestamps: the kernel ABI half of `capture/camera.rs`.
 *
 * Adapted from PR #270 (packages/slam-rs/crates/robocap-recorder/native/camera.c at 271ce643), which was verified on Cap A and
 * Cap B: rkisp mainpath, mplane API, NV12 1920x1080 in ONE plane (stride 1920, chroma after luma), 8 MMAP buffers. Changes:
 * names are prefixed; the buffers can also be MMAP with the non-coherent cache hint, or dma-bufs from a heap (DMABUF); a
 * dequeued buffer is either lent out (rl_camera_dequeue, then rl_camera_queue) or has its LUMA plane only (1920*1080 bytes)
 * copied out before it is requeued (rl_camera_next).
 * kornia-io's V4L2 capture is single-planar only; this is the natural upstream for it (see UPSTREAM.md). */
#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define RL_WIDTH 1920
#define RL_HEIGHT 1080
#define RL_LUMA (RL_WIDTH * RL_HEIGHT)
#define RL_NV12 (RL_LUMA * 3 / 2)
#define RL_BUFFERS 8

/* Linux 6.0 split `struct v4l2_requestbuffers.reserved[0]` into `__u8 flags; __u8 reserved[3]`; the sysroot headers are 5.14.
 * On little-endian aarch64 the flags byte is the low byte of reserved[0]. */
#ifndef V4L2_MEMORY_FLAG_NON_COHERENT
#define V4L2_MEMORY_FLAG_NON_COHERENT (1 << 0)
#endif

/* linux/dma-heap.h is Linux 5.6; the host (x86_64) sysroot headers are 4.18. */
#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#else
struct dma_heap_allocation_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)
#endif

/* Buffer memory modes (camera.rs `CaptureMemory`). */
#define RL_MEMORY_MMAP 0
#define RL_MEMORY_MMAP_NON_COHERENT 1
#define RL_MEMORY_DMABUF 2

/* The step rl_camera_open_with was at when it failed (this thread's last failure), for the error message. */
static _Thread_local const char *rl_open_step = "";
const char *rl_camera_open_step(void) { return rl_open_step; }

struct rl_camera {
    int fd, streaming, memory; /* memory: V4L2_MEMORY_MMAP or V4L2_MEMORY_DMABUF */
    unsigned count;
    void *buffers[RL_BUFFERS];
    size_t lengths[RL_BUFFERS];
    int dmabufs[RL_BUFFERS]; /* heap buffers in DMABUF mode, -1 otherwise */
    uint32_t mmap_capabilities, dmabuf_capabilities;
    uint32_t memory_flags; /* REQBUFS flags as the driver returned them */
    struct v4l2_format original;
};

void rl_camera_close(struct rl_camera *camera) {
    if (!camera) return;
    if (camera->fd >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (camera->streaming) ioctl(camera->fd, VIDIOC_STREAMOFF, &type);
        for (unsigned i = 0; i < camera->count; ++i)
            if (camera->buffers[i]) munmap(camera->buffers[i], camera->lengths[i]);
        struct v4l2_requestbuffers request = {.type = type, .memory = (uint32_t)camera->memory, .count = 0};
        ioctl(camera->fd, VIDIOC_REQBUFS, &request);
        if (camera->original.fmt.pix_mp.width) ioctl(camera->fd, VIDIOC_S_FMT, &camera->original);
        close(camera->fd);
    }
    for (unsigned i = 0; i < RL_BUFFERS; ++i)
        if (camera->dmabufs[i] >= 0) close(camera->dmabufs[i]);
    free(camera);
}

/* Requeue buffer `index` (DMABUF: with its heap fd). 0, or -1 + errno. */
int rl_camera_queue(struct rl_camera *camera, unsigned index) {
    if (index >= camera->count) {
        errno = EINVAL;
        return -1;
    }
    struct v4l2_plane plane = {.length = (uint32_t)camera->lengths[index]};
    if (camera->memory == V4L2_MEMORY_DMABUF) plane.m.fd = camera->dmabufs[index];
    struct v4l2_buffer buffer = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                                 .memory = (uint32_t)camera->memory,
                                 .index = index,
                                 .length = 1,
                                 .m.planes = &plane};
    return ioctl(camera->fd, VIDIOC_QBUF, &buffer);
}

/* Open, negotiate NV12 1920x1080 single-plane mplane, get `count` buffers in `mode` (RL_MEMORY_*; DMABUF allocates them from the
 * dma-buf heap device `heap`), map and queue them, STREAMON. NULL + errno on failure. */
struct rl_camera *rl_camera_open_with(const char *path, int mode, const char *heap, unsigned count) {
    int heap_fd = -1;
    if (count == 0 || count > RL_BUFFERS || mode < RL_MEMORY_MMAP || mode > RL_MEMORY_DMABUF || (mode == RL_MEMORY_DMABUF && !heap)) {
        errno = EINVAL;
        return NULL;
    }
    struct rl_camera *camera = calloc(1, sizeof(*camera));
    if (!camera) return NULL;
    for (unsigned i = 0; i < RL_BUFFERS; ++i) camera->dmabufs[i] = -1;
    camera->memory = mode == RL_MEMORY_DMABUF ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
    rl_open_step = "open";
    camera->fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (camera->fd < 0) goto fail;
    camera->original.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    rl_open_step = "G_FMT/S_FMT";
    if (ioctl(camera->fd, VIDIOC_G_FMT, &camera->original)) goto fail;
    struct v4l2_format format = camera->original;
    format.fmt.pix_mp.width = RL_WIDTH;
    format.fmt.pix_mp.height = RL_HEIGHT;
    format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    format.fmt.pix_mp.field = V4L2_FIELD_NONE;
    if (ioctl(camera->fd, VIDIOC_S_FMT, &format)) goto fail;
    if (format.fmt.pix_mp.width != RL_WIDTH || format.fmt.pix_mp.height != RL_HEIGHT ||
        format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 || format.fmt.pix_mp.num_planes != 1 ||
        format.fmt.pix_mp.plane_fmt[0].bytesperline != RL_WIDTH || format.fmt.pix_mp.plane_fmt[0].sizeimage < RL_NV12) {
        errno = ENOTSUP;
        goto fail;
    }
    /* REQBUFS with count 0 allocates nothing and reports what each memory type supports (an error: the type is refused). */
    struct v4l2_requestbuffers probe = {.type = format.type, .memory = V4L2_MEMORY_MMAP, .count = 0};
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &probe) == 0) camera->mmap_capabilities = probe.capabilities;
    probe = (struct v4l2_requestbuffers){.type = format.type, .memory = V4L2_MEMORY_DMABUF, .count = 0};
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &probe) == 0) camera->dmabuf_capabilities = probe.capabilities;
    struct v4l2_requestbuffers request = {.type = format.type, .memory = (uint32_t)camera->memory, .count = count};
    if (mode == RL_MEMORY_MMAP_NON_COHERENT) request.reserved[0] = V4L2_MEMORY_FLAG_NON_COHERENT;
    rl_open_step = "REQBUFS";
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &request)) goto fail;
    camera->memory_flags = request.reserved[0] & 0xff;
    if (!request.count || request.count > RL_BUFFERS) {
        errno = EOVERFLOW;
        goto fail;
    }
    camera->count = request.count;
    size_t image_bytes = format.fmt.pix_mp.plane_fmt[0].sizeimage;
    if (mode == RL_MEMORY_DMABUF) {
        rl_open_step = "open heap";
        heap_fd = open(heap, O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) goto fail;
    }
    for (unsigned i = 0; i < camera->count; ++i) {
        struct v4l2_plane plane = {0};
        struct v4l2_buffer buffer = {
            .type = format.type, .memory = (uint32_t)camera->memory, .index = i, .length = 1, .m.planes = &plane};
        rl_open_step = "QUERYBUF";
        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &buffer)) goto fail;
        void *address;
        if (mode == RL_MEMORY_DMABUF) {
            /* QUERYBUF reports the driver's minimum plane length, which can exceed sizeimage. */
            size_t bytes = plane.length > image_bytes ? plane.length : image_bytes;
            struct dma_heap_allocation_data allocation = {.len = bytes, .fd_flags = O_RDWR | O_CLOEXEC};
            rl_open_step = "DMA_HEAP_IOCTL_ALLOC";
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation)) goto fail;
            camera->dmabufs[i] = (int)allocation.fd;
            camera->lengths[i] = bytes;
            address = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, camera->dmabufs[i], 0);
        } else {
            camera->lengths[i] = plane.length;
            address = mmap(NULL, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, camera->fd, plane.m.mem_offset);
        }
        rl_open_step = "mmap";
        if (address == MAP_FAILED) goto fail;
        camera->buffers[i] = address;
        rl_open_step = "QBUF";
        if (rl_camera_queue(camera, i)) goto fail;
    }
    if (heap_fd >= 0) close(heap_fd);
    heap_fd = -1;
    rl_open_step = "STREAMON";
    if (ioctl(camera->fd, VIDIOC_STREAMON, &format.type)) goto fail;
    camera->streaming = 1;
    return camera;
fail: {
    int saved = errno;
    if (heap_fd >= 0) close(heap_fd);
    rl_camera_close(camera);
    errno = saved;
    return NULL;
}
}

/* The production setup: 8 driver (MMAP) buffers, default cache mode. */
struct rl_camera *rl_camera_open(const char *path) { return rl_camera_open_with(path, RL_MEMORY_MMAP, NULL, RL_BUFFERS); }

/* REQBUFS capabilities of MMAP and DMABUF (0 = that memory type was refused), the REQBUFS flags the driver kept, the buffer count,
 * the bytes of buffer 0. */
void rl_camera_info(const struct rl_camera *camera, uint32_t *mmap_capabilities, uint32_t *dmabuf_capabilities,
                    uint32_t *memory_flags, unsigned *count, size_t *buffer_bytes) {
    *mmap_capabilities = camera->mmap_capabilities;
    *dmabuf_capabilities = camera->dmabuf_capabilities;
    *memory_flags = camera->memory_flags;
    *count = camera->count;
    *buffer_bytes = camera->lengths[0];
}

/* Wait up to timeout_ms for a frame and lend its buffer out: *index, *luma (the mapped plane at its data offset, RL_NV12 bytes
 * valid), the V4L2 metadata. The caller requeues it with rl_camera_queue. Returns 1 for a frame, 0 for a timeout, -1 for a fault
 * (errno set; EIO = a buffer that failed validation, already requeued). */
int rl_camera_dequeue(struct rl_camera *camera, unsigned *index, const unsigned char **luma, int64_t *timestamp_ns,
                      uint32_t *sequence, uint32_t *flags, int timeout_ms) {
    struct pollfd poll_fd = {.fd = camera->fd, .events = POLLIN};
    int ready = poll(&poll_fd, 1, timeout_ms);
    if (ready <= 0) return ready;
    struct v4l2_plane plane = {0};
    struct v4l2_buffer buffer = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, .memory = (uint32_t)camera->memory, .length = 1, .m.planes = &plane};
    if (ioctl(camera->fd, VIDIOC_DQBUF, &buffer)) return -1;
    if (buffer.index >= camera->count || plane.data_offset > plane.bytesused ||
        plane.bytesused > camera->lengths[buffer.index] || plane.bytesused - plane.data_offset != RL_NV12 ||
        (buffer.flags & V4L2_BUF_FLAG_ERROR)) {
        if (buffer.index < camera->count && rl_camera_queue(camera, buffer.index)) return -1;
        errno = EIO;
        return -1;
    }
    *index = buffer.index;
    *luma = (const unsigned char *)camera->buffers[buffer.index] + plane.data_offset;
    *timestamp_ns = (int64_t)buffer.timestamp.tv_sec * 1000000000 + (int64_t)buffer.timestamp.tv_usec * 1000;
    *sequence = buffer.sequence;
    *flags = buffer.flags;
    return 1;
}

/* Wait up to timeout_ms for a frame; copy its luma plane (RL_LUMA bytes) to `luma`, requeue the buffer.
 * Returns 1 for a frame, 0 for a timeout, -1 for a fault (errno set; EIO = a buffer that failed validation). */
int rl_camera_next(struct rl_camera *camera, unsigned char *luma, size_t capacity, int64_t *timestamp_ns,
                   uint32_t *sequence, uint32_t *flags, int timeout_ms) {
    unsigned index = 0;
    const unsigned char *frame = NULL;
    int result = rl_camera_dequeue(camera, &index, &frame, timestamp_ns, sequence, flags, timeout_ms);
    if (result <= 0) return result;
    if (capacity >= RL_LUMA) memcpy(luma, frame, RL_LUMA);
    if (rl_camera_queue(camera, index)) return -1;
    if (capacity < RL_LUMA) {
        errno = EIO;
        return -1;
    }
    return 1;
}

/* VIDIOC_EXPBUF: a new dma-buf fd (O_RDWR | O_CLOEXEC) for buffer `index`, or -1 + errno. */
int rl_camera_export(const struct rl_camera *camera, unsigned index) {
    struct v4l2_exportbuffer request = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, .index = index, .plane = 0, .flags = O_RDWR | O_CLOEXEC};
    if (ioctl(camera->fd, VIDIOC_EXPBUF, &request)) return -1;
    return request.fd;
}

/* The heap dma-buf of buffer `index` in DMABUF mode, else -1. */
int rl_camera_dmabuf(const struct rl_camera *camera, unsigned index) {
    return index < camera->count ? camera->dmabufs[index] : -1;
}

/* DMA_BUF_IOCTL_SYNC on a dma-buf fd: begin (start != 0) or end (start == 0) a CPU read. 0, or -1 + errno. */
int rl_dmabuf_sync_read(int fd, int start) {
    struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_READ | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END)};
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}
