// Keep dav1d's C structs behind this bridge; Rust owns only an opaque decoder.
// The library is loaded at runtime so H.264 replay does not require dav1d.
#include <dav1d/dav1d.h>
#include <dav1d/version.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if DAV1D_API_VERSION_MAJOR != 6
#error "catalog replay requires dav1d 1.2.x headers (libdav1d.so.6 ABI)"
#endif

typedef struct {
    void *lib;
    Dav1dContext *ctx;
    Dav1dPicture pic;
    void (*settings)(Dav1dSettings *);
    int (*open)(Dav1dContext **, const Dav1dSettings *);
    uint8_t *(*create)(Dav1dData *, size_t);
    int (*send)(Dav1dContext *, Dav1dData *);
    int (*get)(Dav1dContext *, Dav1dPicture *);
    void (*unref)(Dav1dPicture *);
    void (*data_unref)(Dav1dData *);
    void (*close)(Dav1dContext **);
} Decoder;

void *catalog_av1_open(const char *path) {
    Decoder *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!d->lib) {
        fprintf(stderr, "dav1d: %s\n", dlerror());
        free(d);
        return NULL;
    }
#define LOAD(field, name) do { \
    *(void **)(&d->field) = dlsym(d->lib, name); \
    if (!d->field) { dlclose(d->lib); free(d); return NULL; } \
} while (0)
    LOAD(settings, "dav1d_default_settings");
    LOAD(open, "dav1d_open");
    LOAD(create, "dav1d_data_create");
    LOAD(send, "dav1d_send_data");
    LOAD(get, "dav1d_get_picture");
    LOAD(unref, "dav1d_picture_unref");
    LOAD(data_unref, "dav1d_data_unref");
    LOAD(close, "dav1d_close");
#undef LOAD
    Dav1dSettings settings;
    d->settings(&settings);
    settings.n_threads = 1;
    settings.max_frame_delay = 1;
    if (d->open(&d->ctx, &settings)) {
        dlclose(d->lib);
        free(d);
        return NULL;
    }
    return d;
}

int catalog_av1_send(void *decoder, const uint8_t *bytes, size_t length) {
    Decoder *d = decoder;
    Dav1dData data = {0};
    uint8_t *target = d->create(&data, length);
    if (!target) return -ENOMEM;
    memcpy(target, bytes, length);
    int status = d->send(d->ctx, &data);
    d->data_unref(&data);
    return status;
}

int catalog_av1_get(void *decoder, const uint8_t **bytes, int *width,
                   int *height, ptrdiff_t *stride, int *full_range) {
    Decoder *d = decoder;
    d->unref(&d->pic);
    int status = d->get(d->ctx, &d->pic);
    if (status) return status;
    if (d->pic.p.bpc != 8) return -EINVAL;
    *full_range = d->pic.seq_hdr->color_range;
    *bytes = d->pic.data[0];
    *width = d->pic.p.w;
    *height = d->pic.p.h;
    *stride = d->pic.stride[0];
    return 0;
}

void catalog_av1_close(void *decoder) {
    Decoder *d = decoder;
    d->unref(&d->pic);
    d->close(&d->ctx);
    dlclose(d->lib);
    free(d);
}
