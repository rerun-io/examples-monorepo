//! Headless wgpu GPU context for standalone rendering.
//!
//! Initializes a GPU device and queue without requiring a window or surface.
//! Used by the `gsplat-render` CLI for headless Gaussian splat rendering.

/// Headless GPU context holding a wgpu device and queue.
pub struct GpuContext {
    /// The GPU device for creating resources and submitting work.
    pub device: wgpu::Device,
    /// The command queue for submitting GPU work.
    pub queue: wgpu::Queue,
    /// Adapter info (GPU name, backend, etc.) for diagnostics.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new headless GPU context.
    ///
    /// Selects a high-performance GPU adapter and requests device limits
    /// sufficient for the Gaussian splat compute pipeline (11 storage
    /// buffers per shader stage, large buffer sizes).
    ///
    /// No window or surface is needed — all rendering is to storage textures.
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| anyhow::anyhow!("No GPU adapter found: {e}"))?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gsplat-render"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage:
                    adapter_limits.max_storage_buffers_per_shader_stage.min(11),
                max_compute_workgroups_per_dimension:
                    adapter_limits.max_compute_workgroups_per_dimension,
                max_buffer_size: adapter_limits.max_buffer_size,
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_compute_invocations_per_workgroup:
                    adapter_limits.max_compute_invocations_per_workgroup,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        }))
        .map_err(|e| anyhow::anyhow!("Failed to create device: {e}"))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }
}
