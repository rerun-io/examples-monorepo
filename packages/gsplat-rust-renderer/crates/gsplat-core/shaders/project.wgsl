// Brush project_forward + project_visible, adapted to raw buffers.
@group(0) @binding(1) var<storage,read> transforms: array<f32>;
@group(0) @binding(2) var<storage,read> raw_opacity: array<f32>;
@group(0) @binding(3) var<storage,read> min_scale: array<f32>;
@group(0) @binding(4) var<storage,read_write> ids: array<u32>;
@group(0) @binding(5) var<storage,read_write> depths: array<u32>;
@group(0) @binding(6) var<storage,read_write> counts: array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> hits: array<u32>;
@group(0) @binding(8) var<storage,read_write> projected: array<Splat>;
@group(0) @binding(9) var<storage,read> coeffs: array<f32>;
struct Projection { valid: bool, xy:vec2f, conic:vec3f, opacity:f32, depth:f32, mean:vec3f }
fn project(id: u32) -> Projection {
    let base = id*10u;
    let mean = vec3f(transforms[base],transforms[base+1u],transforms[base+2u]);
    let point = (u.view*vec4f(mean,1.0)).xyz;
    var result: Projection;
    if !finite3(point) || point.z > 1e10 || point.z < 0.01 { return result; }
    var scale = exp(vec3f(transforms[base+7u],transforms[base+8u],transforms[base+9u])+u.options.x);
    if !finite3(scale) { return result; }
    let qraw = vec4f(transforms[base+3u],transforms[base+4u],transforms[base+5u],transforms[base+6u]);
    let qnorm = dot(qraw,qraw);
    if !(qnorm >= 1e-6 && finite(qnorm)) || !finite(raw_opacity[id]) { return result; }
    var opacity = 1.0/(1.0+exp(-raw_opacity[id]));
    if u.options.z != 0.0 {
        let floor = min_scale[id];
        let filtered = sqrt(scale*scale+floor*floor);
        let ratio = scale/filtered;
        opacity = clamp(opacity*(ratio.x*ratio.y*ratio.z),1e-6,1.0-1e-6);
        scale = filtered;
    }
    let q = qraw*(1.0/sqrt(qnorm));
    let w=q.x; let x=q.y; let y=q.z; let z=q.w;
    let rotation = mat3x3f(vec3f(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y)),
                          vec3f(2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x)),
                          vec3f(2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y)));
    let vr = mat3x3f(u.view[0].xyz,u.view[1].xyz,u.view[2].xyz)*rotation;
    let ns = mat3x3f(vr[0]*scale.x,vr[1]*scale.y,vr[2]*scale.z);
    let inv_z = 1.0/point.z;
    let d = u.pinhole.xy*inv_z;
    let clamped = clamp(point.xy*inv_z,u.clamp_limits.xy,u.clamp_limits.zw);
    let jx = vec3f(d.x,0.0,-d.x*clamped.x);
    let jy = vec3f(0.0,d.y,-d.y*clamped.y);
    let v0 = vec3f(dot(jx,ns[0]),dot(jx,ns[1]),dot(jx,ns[2]));
    let v1 = vec3f(dot(jy,ns[0]),dot(jy,ns[1]),dot(jy,ns[2]));
    var cov = vec3f(dot(v0,v0),dot(v0,v1),dot(v1,v1));
    let max_abs = max(max(abs(cov.x),abs(cov.y)),abs(cov.z));
    cov *= select(1.0,1e18/max_abs,max_abs>1e18);
    let raw_cov = cov;
    let blur = select(0.3,0.1,u.options.y != 0.0);
    cov += vec3f(blur,0.0,blur);
    if u.options.y != 0.0 {
        let raw_det = raw_cov.x*raw_cov.z-raw_cov.y*raw_cov.y;
        let blurred_det = cov.x*cov.z-cov.y*cov.y;
        opacity *= sqrt(max(raw_det,0.0)/blurred_det);
    }
    if !finite3(cov) { return result; }
    if !(opacity >= 1.0/255.0) { return result; }
    let det = cov.x*cov.z-cov.y*cov.y;
    let conic = vec3f(cov.z,-cov.y,cov.x)*select(0.0,1.0/det,det>0.0);
    let extent = bbox_extent(conic,log(opacity*255.0));
    if !(extent.x >= 0.0 && extent.y >= 0.0) { return result; }
    let xy = u.pinhole.xy*point.xy*inv_z+u.pinhole.zw;
    if !(all(xy+extent>vec2f(0.0)) && all(xy-extent<vec2f(u.image.xy))) { return result; }
    result = Projection(true,xy,conic,opacity,point.z,mean);
    return result;
}
fn coefficient(base:u32, index:u32) -> vec3f {
    let i=base+index*3u; return vec3f(coeffs[i],coeffs[i+1u],coeffs[i+2u]);
}
fn sh_color(id:u32, v:vec3f) -> vec3f {
    let base=id*u.scene.z*3u;
    var color = coefficient(base,0u)*0.2820948;
    if u.scene.y >= 1u {
        color += coefficient(base,1u)*(-0.4886025*v.y);
        color += coefficient(base,2u)*(0.4886025*v.z);
        color += coefficient(base,3u)*(-0.4886025*v.x);
    }
    if u.scene.y >= 2u {
        let z2=v.z*v.z; let fc1=v.x*v.x-v.y*v.y; let fs1=2.0*v.x*v.y;
        let p6=0.9461747*z2-0.31539157;
        color += coefficient(base,4u)*(0.54627424*fs1);
        color += coefficient(base,5u)*(-1.0925485*v.z*v.y);
        color += coefficient(base,6u)*p6;
        color += coefficient(base,7u)*(-1.0925485*v.z*v.x);
        color += coefficient(base,8u)*(0.54627424*fc1);
        if u.scene.y >= 3u {
            let f0c=-2.285229*z2+0.4570458; let f1b=1.4453057*v.z;
            let fc2=v.x*fc1-v.y*fs1; let fs2=v.x*fs1+v.y*fc1;
            let p12=v.z*(1.8658817*z2-1.119529);
            color += coefficient(base,9u)*(-0.5900436*fs2);
            color += coefficient(base,10u)*(f1b*fs1);
            color += coefficient(base,11u)*(f0c*v.y);
            color += coefficient(base,12u)*p12;
            color += coefficient(base,13u)*(f0c*v.x);
            color += coefficient(base,14u)*(f1b*fc1);
            color += coefficient(base,15u)*(-0.5900436*fc2);
            if u.scene.y >= 4u {
                let f0d=v.z*(-4.683326*z2+2.0071396); let f1c=3.3116114*z2-0.47308735;
                let f2b=-1.7701308*v.z; let fc3=v.x*fc2-v.y*fs2; let fs3=v.x*fs2+v.y*fc2;
                color += coefficient(base,16u)*(0.62583575*fs3);
                color += coefficient(base,17u)*(f2b*fs2);
                color += coefficient(base,18u)*(f1c*fs1);
                color += coefficient(base,19u)*(f0d*v.y);
                color += coefficient(base,20u)*(1.9843135*v.z*p12-1.0062306*p6);
                color += coefficient(base,21u)*(f0d*v.x);
                color += coefficient(base,22u)*(f1c*fc1);
                color += coefficient(base,23u)*(f2b*fc2);
                color += coefficient(base,24u)*(0.62583575*fc3);
            }
        }
    }
    color += 0.5;
    return clamp(vec3f(select(0.0,color.x,finite(color.x)),select(0.0,color.y,finite(color.y)),select(0.0,color.z,finite(color.z))),vec3f(-100.0),vec3f(100.0));
}
@compute @workgroup_size(256)
fn project_forward(@builtin(workgroup_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lid:u32) {
    let id=(gid.x+gid.y*groups.x)*256u+lid;
    if id>=u.scene.x { return; }
    let p=project(id);
    if !p.valid { return; }
    let power=log(p.opacity*255.0); let bb=tile_bbox(p.xy,bbox_extent(p.conic,power));
    let width=bb.z-bb.x; let n=(bb.w-bb.y)*width;
    var hit=0u;
    for (var i=0u;i<n;i++) {
        let tile=vec2u(i%width+bb.x,i/width+bb.y);
        if tile_hit(tile,p.xy,p.conic,power) { hit++; }
    }
    hits[id]=hit;
    atomicAdd(&counts[1],hit);
    let write=atomicAdd(&counts[0],1u);
    ids[write]=id; depths[write]=bitcast<u32>(p.depth);
}
@compute @workgroup_size(256)
fn project_visible(@builtin(workgroup_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lid:u32) {
    let compact=(gid.x+gid.y*groups.x)*256u+lid;
    if compact>=atomicLoad(&counts[0]) { return; }
    let id=ids[compact]; let p=project(id);
    let v=normalize(p.mean-u.camera.xyz); let color=sh_color(id,v);
    projected[compact]=Splat(p.xy.x,p.xy.y,p.conic.x,p.conic.y,p.conic.z,p.opacity,color.x,color.y,color.z);
}
