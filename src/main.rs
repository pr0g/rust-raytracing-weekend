use sdl2::event::Event;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

use std::error::Error;
use std::time::Duration;
use std::fs::{ OpenOptions };
use std::io::{ BufWriter, Write };

fn create_ppm_file(/*out: &mut Write*/) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .open("ppm-test.ppm");

    let mut writer = match file {
        Ok(file) => BufWriter::new(file),
        Err(e) => return Err(e)
    };

    let width = 200;
    let height = 100;
    writer.write_fmt(format_args!("P3\n{} {}\n255\n", width, height))?;

    for y in (0..=height-1).rev() {
        for x in 0..=width-1 {
            let (r, g, b) = (x as f32 / width as f32, y as f32 / height as f32, 0.2_f32);
            let (ir, ig, ib) = ((255_f32 * r) as i32, (255_f32 * g) as i32, (255_f32 * b) as i32);
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

    'running:loop {
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
