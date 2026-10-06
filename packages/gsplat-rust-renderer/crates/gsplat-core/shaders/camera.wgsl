// Lens projection/Jacobians hand-ported from Brush camera_model at 1388f74c.
// The default value uses a uniform switch; fixed overrides specialize the same source.
override CAMERA_MODEL:u32=0xffffffffu;
fn camera_kind()->u32 {if CAMERA_MODEL!=0xffffffffu{return CAMERA_MODEL;}return u.lens.x;}
struct CameraProjection {xy:vec2f,jx:vec3f,jy:vec3f}
fn project_camera(point:vec3f)->CameraProjection {
    let fx=u.pinhole.x;let fy=u.pinhole.y;
    let x=point.x;let y=point.y;let z=point.z;let iz=1.0/z;
    let kind=camera_kind();
    let k=u.coeff0;let e=u.coeff1;
    if kind==0u {
        let d=u.pinhole.xy*iz;let clamped=clamp(point.xy*iz,u.clamp_limits.xy,u.clamp_limits.zw);
        return CameraProjection(u.pinhole.xy*point.xy*iz+u.pinhole.zw,vec3f(d.x,0.0,-d.x*clamped.x),vec3f(0.0,d.y,-d.y*clamped.y));
    }
    if kind==2u {
        let p1=e.z;let p2=e.w;
        let v=point.xy/z;let r2=dot(v,v);let r4=r2*r2;let r6=r4*r2;
        let radial=(1.0+k.x*r2+k.y*r4+k.z*r6)/(1.0+k.w*r2+e.x*r4+e.y*r6);
        let xy=u.pinhole.xy*(v*radial+vec2f(2.0*p1*v.x*v.y+p2*(r2+2.0*v.x*v.x),2.0*p2*v.x*v.y+p1*(r2+2.0*v.y*v.y)))+u.pinhole.zw;
        let bounded=clamp(point.xy*iz,u.clamp_limits.xy,u.clamp_limits.zw);let rb=length(bounded);
        let normalized=bounded*select(1.0,u.camera_limits.y/rb,rb>u.camera_limits.y);
        let xn=normalized.x;let yn=normalized.y;let xc=xn*z;let yc=yn*z;
        let a=xn*xn+yn*yn;let a2=a*a;let a3=a2*a;
        let num=1.0+k.x*a+k.y*a2+k.z*a3;let den=1.0+k.w*a+e.x*a2+e.y*a3;
        let np=k.x+2.0*k.y*a+3.0*k.z*a2;let dp=k.w+2.0*e.x*a+3.0*e.y*a2;
        let inv=1.0/den;let r=num*inv;let rp=(np*den-num*dp)*(inv*inv);
        let d00=r+2.0*xn*xn*rp+2.0*p1*yn+6.0*p2*xn;
        let d01=2.0*xn*yn*rp+2.0*p1*xn+2.0*p2*yn;
        let d11=r+2.0*yn*yn*rp+6.0*p1*yn+2.0*p2*xn;
        return CameraProjection(xy,vec3f(fx*d00*iz,fx*d01*iz,-fx*(d00*xc+d01*yc)*(iz*iz)),vec3f(fy*d01*iz,fy*d11*iz,-fy*(d01*xc+d11*yc)*(iz*iz)));
    }
    let x2=x*x;let y2=y*y;let xy=x*y;let r2=x2+y2;let r=sqrt(r2);
    var result=CameraProjection(u.pinhole.xy*point.xy*iz+u.pinhole.zw,vec3f(fx*iz,0.0,-fx*iz*x*iz),vec3f(0.0,fy*iz,-fy*iz*y*iz));
    if r>=1e-6 {
        let theta=atan2(r,z);let t2=theta*theta;let t4=t2*t2;let t6=t2*t4;let t8=t4*t4;
        let d=theta*(1.0+k.x*t2+k.y*t4+k.z*t6+k.w*t8);
        let deriv=1.0+3.0*k.x*t2+5.0*k.y*t4+7.0*k.z*t6+9.0*k.w*t8;
        let ir=1.0/r;let ir3=ir*ir*ir;let irho=1.0/(r2+z*z);let irhor=irho*ir;
        let dd=deriv*vec3f(x*z*irhor,y*z*irhor,-r*irho);
        let xr=x*ir;let yr=y*ir;
        result.xy=u.pinhole.xy*(d*point.xy*ir)+u.pinhole.zw;
        result.jx=fx*vec3f(dd.x*xr+d*y2*ir3,dd.y*xr-d*xy*ir3,dd.z*xr);
        result.jy=fy*vec3f(dd.x*yr-d*xy*ir3,dd.y*yr+d*x2*ir3,dd.z*yr);
    }
    if kind==3u {
        let p1=e.x;let p2=e.y;let sx=e.z;let sy=e.w;
        let nu=2.0*p1*xy+p2*(3.0*x2+y2)+sx*r2;
        let nv=2.0*p2*xy+p1*(x2+3.0*y2)+sy*r2;
        let nux=2.0*(p1*y+(3.0*p2+sx)*x);let nuy=2.0*(p1*x+(p2+sx)*y);
        let nvx=2.0*(p2*y+(p1+sy)*x);let nvy=2.0*(p2*x+(3.0*p1+sy)*y);
        let iz2=iz*iz;let iz3=iz2*iz;
        result.xy+=u.pinhole.xy*vec2f(nu,nv)*iz2;
        result.jx+=vec3f(fx*nux*iz2,fx*nuy*iz2,-2.0*fx*nu*iz3);
        result.jy+=vec3f(fy*nvx*iz2,fy*nvy*iz2,-2.0*fy*nv*iz3);
    }
    return result;
}
