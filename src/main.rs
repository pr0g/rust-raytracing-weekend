use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;

use std::error::Error;
use std::f32;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Duration;

struct Hitables {
    hitables: Vec<Box<dyn Hitable>>,
}

impl Hitables {
    fn new() -> Self {
        Hitables {
            hitables: Vec::new(),
        }
    }
}

impl Hitable for Hitables {
    fn hit(self: &Self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let mut closest_hit: Option<Hit> = None;
        for hitable in self.hitables.iter() {
            match hitable.hit(ray, t_min, t_max) {
                Some(hit) => match closest_hit {
                    Some(closest) => {
                        if hit.t < closest.t {
                            closest_hit = Some(closest);
                        }
                    }
                    None => closest_hit = Some(hit),
                },
                _ => continue,
            }
        }

        closest_hit
    }
}

trait Hitable {
    fn hit(self: &Self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit>;
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    t: f32,
    p: Vec3,
    normal: Vec3,
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = &ray.origin - &self.center;
        let a = length_sq(&ray.direction);
        let b = dot(&oc, &ray.direction);
        let c = length_sq(&oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant > 0f32 {
            let temp = (-b - discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                return Some(Hit {
                    t: temp,
                    p: ray.point_at_t(temp),
                    normal: &(&ray.point_at_t(temp) - &self.center) / self.radius,
                });
            }
            let temp = (-b + discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                return Some(Hit {
                    t: temp,
                    p: ray.point_at_t(temp),
                    normal: &(&ray.point_at_t(temp) - &self.center) / self.radius,
                });
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Sphere {
    fn new(center: &Vec3, radius: f32) -> Self {
        Sphere {
            center: *center,
            radius,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: &Vec3, direction: &Vec3) -> Self {
        Ray {
            origin: *origin,
            direction: *direction,
        }
    }

    fn point_at_t(&self, t: f32) -> Vec3 {
        &self.origin + &(&self.direction * t)
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn zero() -> Self {
        Vec3 {
            ..Default::default()
        }
    }

    fn one() -> Self {
        Vec3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        }
    }

    fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
}

impl Neg for &Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Add for &Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for &Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for &Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<f32> for &Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

fn dot(lhs: &Vec3, rhs: &Vec3) -> f32 {
    lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z
}

fn length(vec: &Vec3) -> f32 {
    length_sq(vec).sqrt()
}

fn length_sq(vec: &Vec3) -> f32 {
    dot(vec, vec)
}

fn normalize(vec: &Vec3) -> Vec3 {
    vec / length(&vec)
}

fn cross(lhs: &Vec3, rhs: &Vec3) -> Vec3 {
    Vec3 {
        x: lhs.y * rhs.z - lhs.z * rhs.y,
        y: lhs.x * rhs.z - lhs.z * rhs.x,
        z: lhs.x * rhs.y - lhs.y * rhs.x,
    }
}

fn create_ppm_file() -> std::io::Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .open("ppm-test.ppm");

    let mut writer = match file {
        Ok(file) => BufWriter::new(file),
        Err(e) => return Err(e),
    };

    let width = 200;
    let height = 100;
    writer.write_fmt(format_args!("P3\n{} {}\n255\n", width, height))?;

    let lower_left_corner = Vec3::new(-2_f32, -1_f32, -1_f32);
    let horizontal = Vec3::new(4f32, 0f32, 0f32);
    let vertical = Vec3::new(0f32, 2f32, 0f32);
    let origin = Vec3::zero();

    let mut hitables = Hitables::new();
    hitables
        .hitables
        .push(Box::new(Sphere::new(&Vec3::new(0f32, 0f32, -1f32), 0.5f32)));
    hitables.hitables.push(Box::new(Sphere::new(
        &Vec3::new(0f32, -100.5f32, -1f32),
        100f32,
    )));

    for y in (0..height).rev() {
        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;
            let ray = Ray::new(
                &origin,
                &(&(&lower_left_corner + &(&horizontal * u)) + &(&vertical * v)),
            );

            let col = match hitables.hit(&ray, 0f32, f32::MAX) {
                Some(hit) => {
                    &Vec3::new(
                        hit.normal.x + 1f32,
                        hit.normal.y + 1f32,
                        hit.normal.z + 1f32,
                    ) * 0.5f32
                }
                None => {
                    let unit_dir = normalize(&ray.direction);
                    let t = 0.5f32 * unit_dir.y + 1f32;
                    &(&Vec3::one() * (1f32 - t)) + &(&Vec3::new(0.5f32, 0.7f32, 1f32) * t)
                }
            };

            let (ir, ig, ib) = (
                (255_f32 * col.x) as i32,
                (255_f32 * col.y) as i32,
                (255_f32 * col.z) as i32,
            );

            writer.write_fmt(format_args!("{} {} {}\n", ir, ig, ib))?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<Error>> {
    create_ppm_file()?;

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("rust-sdl2", 800, 600)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().build()?;

    let cornflower_blue = Color::RGB(100, 149, 237);
    canvas.set_draw_color(cornflower_blue);
    canvas.clear();
    canvas.present();

    let mut event_pump = sdl_context.event_pump()?;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::cross;
    use super::dot;
    use super::f32;
    use super::length;
    use super::length_sq;
    use super::normalize;
    use super::Hitable;
    use super::Hitables;
    use super::Ray;
    use super::Sphere;
    use super::Vec3;

    #[test]
    fn vec3_zero() {
        let zero = Vec3::zero();
        assert_eq!(zero.x, 0_f32);
        assert_eq!(zero.y, 0_f32);
        assert_eq!(zero.z, 0_f32);
    }

    #[test]
    fn vec3_add() {
        let lhs = Vec3::new(1_f32, 2_f32, 3_f32);
        let rhs = Vec3::new(4_f32, 5_f32, 6_f32);

        let result = &lhs + &rhs;
        assert_eq!(result.x, 5_f32);
        assert_eq!(result.y, 7_f32);
        assert_eq!(result.z, 9_f32);
    }

    #[test]
    fn vec3_neg() {
        let vec = Vec3 {
            x: 1_f32,
            y: 2_f32,
            z: 3_f32,
        };
        let result = -&vec;
        assert_eq!(result.x, -vec.x);
        assert_eq!(result.y, -vec.y);
        assert_eq!(result.z, -vec.z);
    }

    #[test]
    fn vec3_sub() {
        let lhs = Vec3::new(4_f32, 5_f32, 6_f32);
        let rhs = Vec3::new(1_f32, 4_f32, 2_f32);

        let result = &lhs - &rhs;
        assert_eq!(result.x, 3_f32);
        assert_eq!(result.y, 1_f32);
        assert_eq!(result.z, 4_f32);
    }

    #[test]
    fn vec3_mul() {
        let vec = Vec3 {
            x: 1_f32,
            y: 2_f32,
            z: 3_f32,
        };
        let result = &vec * 2_f32;

        assert_eq!(result.x, 2_f32);
        assert_eq!(result.y, 4_f32);
        assert_eq!(result.z, 6_f32);
    }

    #[test]
    fn vec3_div() {
        let vec = Vec3 {
            x: 8_f32,
            y: 20_f32,
            z: 50_f32,
        };
        let result = &vec / 2_f32;

        assert_eq!(result.x, 4_f32);
        assert_eq!(result.y, 10_f32);
        assert_eq!(result.z, 25_f32);
    }

    #[test]
    fn vec3_dot_perpendicular() {
        let lhs = Vec3::new(1_f32, 0_f32, 0_f32);
        let rhs = Vec3::new(0_f32, 1_f32, 0_f32);

        let dot_result = dot(&lhs, &rhs);
        assert_eq!(dot_result, 0_f32);
    }

    #[test]
    fn vec3_dot_parallel() {
        let lhs = Vec3::new(1_f32, 0_f32, 0_f32);
        let rhs = Vec3::new(1_f32, 0_f32, 0_f32);

        let dot_result = dot(&lhs, &rhs);
        assert_eq!(dot_result, 1_f32);
    }

    #[test]
    fn vec3_length_sq() {
        let vec = Vec3::new(3_f32, 4_f32, 0_f32);
        let len = length_sq(&vec);

        assert_eq!(len, 25_f32);

        let vec = Vec3::new(0_f32, 6_f32, 8_f32);
        let len = length_sq(&vec);

        assert_eq!(len, 100_f32);
    }

    #[test]
    fn vec3_length() {
        {
            let vec = Vec3::new(3_f32, 4_f32, 0_f32);
            let len = length(&vec);

            assert_eq!(len, 5_f32);

            let vec = Vec3::new(0_f32, 6_f32, 8_f32);
            let len = length(&vec);

            assert_eq!(len, 10_f32);
        }
    }

    #[test]
    fn vec3_normalize() {
        let vec = Vec3::new(10_f32, 0_f32, 0_f32);
        let norm_vec = normalize(&vec);

        assert_eq!(norm_vec.x, 1_f32);
        assert_eq!(norm_vec.y, 0_f32);
        assert_eq!(norm_vec.z, 0_f32);

        let vec = Vec3::new(5_f32, 0_f32, 5_f32);
        let norm_vec = normalize(&vec);

        assert_eq!(norm_vec.x, 0.70710677_f32);
        assert_eq!(norm_vec.y, 0_f32);
        assert_eq!(norm_vec.z, 0.70710677_f32);
    }

    #[test]
    fn vec3_cross() {
        let lhs = Vec3::new(1_f32, 0_f32, 0_f32);
        let rhs = Vec3::new(0_f32, 1_f32, 0_f32);
        let cross_result = cross(&lhs, &rhs);

        assert_eq!(cross_result.x, 0_f32);
        assert_eq!(cross_result.y, 0_f32);
        assert_eq!(cross_result.z, 1_f32);

        let lhs = Vec3::new(0_f32, 1_f32, 0_f32);
        let rhs = Vec3::new(1_f32, 0_f32, 0_f32);
        let cross_result = cross(&lhs, &rhs);

        assert_eq!(cross_result.x, 0_f32);
        assert_eq!(cross_result.y, 0_f32);
        assert_eq!(cross_result.z, -1_f32);

        let lhs = Vec3::new(0_f32, 1_f32, 0_f32);
        let rhs = Vec3::new(0_f32, 0_f32, 1_f32);
        let cross_result = cross(&lhs, &rhs);

        assert_eq!(cross_result.x, 1_f32);
        assert_eq!(cross_result.y, 0_f32);
        assert_eq!(cross_result.z, 0_f32);

        let lhs = Vec3::new(1_f32, 0_f32, 0_f32);
        let rhs = Vec3::new(0_f32, 0_f32, 1_f32);
        let cross_result = cross(&lhs, &rhs);

        assert_eq!(cross_result.x, 0_f32);
        assert_eq!(cross_result.y, 1_f32);
        assert_eq!(cross_result.z, 0_f32);
    }

    #[test]
    fn ray_create() {
        let ray = Ray::new(&Vec3::zero(), &Vec3::new(1_f32, 0_f32, 0_f32));

        assert_eq!(ray.origin.x, 0_f32);
        assert_eq!(ray.origin.y, 0_f32);
        assert_eq!(ray.origin.z, 0_f32);

        assert_eq!(ray.direction.x, 1_f32);
        assert_eq!(ray.direction.y, 0_f32);
        assert_eq!(ray.direction.z, 0_f32);
    }

    #[test]
    fn ray_point_at_t() {
        let ray = Ray::new(&Vec3::zero(), &Vec3::new(10_f32, 0_f32, 0_f32));
        let halfway_point = ray.point_at_t(0.5_f32);

        assert_eq!(halfway_point.x, 5_f32);
        assert_eq!(halfway_point.y, 0_f32);
        assert_eq!(halfway_point.z, 0_f32);

        let end_point = ray.point_at_t(1_f32);

        assert_eq!(end_point.x, 10_f32);
        assert_eq!(end_point.y, 0_f32);
        assert_eq!(end_point.z, 0_f32);

        let ray = Ray::new(&Vec3::zero(), &Vec3::new(20_f32, 0_f32, 20_f32));
        let halfway_point = ray.point_at_t(0.5_f32);

        assert_eq!(halfway_point.x, 10_f32);
        assert_eq!(halfway_point.y, 0_f32);
        assert_eq!(halfway_point.z, 10_f32);
    }

    #[test]
    fn ray_hit_sphere() {
        let sphere = Sphere::new(&Vec3::new(0f32, 0f32, -1f32), 0.5f32);
        let hit = sphere.hit(
            &Ray::new(&Vec3::zero(), &Vec3::new(0f32, 0f32, -1f32)),
            0f32,
            f32::MAX,
        );
        assert!(hit.is_some());
    }

    #[test]
    fn ray_miss_sphere() {
        let sphere = Sphere::new(&Vec3::new(0f32, 0f32, -1f32), 0.5f32);
        let hit = sphere.hit(
            &Ray::new(&Vec3::new(0f32, 1f32, 0f32), &Vec3::new(0f32, 0f32, -1f32)),
            0f32,
            f32::MAX,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_hitables() {
        let mut hitables = Hitables::new();
        hitables
            .hitables
            .push(Box::new(Sphere::new(&Vec3::new(0f32, 0f32, -5f32), 0.5f32)));
        hitables.hitables.push(Box::new(Sphere::new(
            &Vec3::new(0f32, 0f32, -10f32),
            0.5f32,
        )));

        let ray = Ray::new(&Vec3::new(0f32, 0f32, 0f32), &Vec3::new(0f32, 0f32, -1f32));
        let hit = hitables.hit(&ray, 0f32, f32::MAX);

        assert!(hit.is_some());
        assert_eq!(hit.unwrap().t, 4.5f32);
    }
}
