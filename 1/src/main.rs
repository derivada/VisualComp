// Uncomment these following global attributes to silence most warnings of "low" interest:
/*
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]
*/
extern crate nalgebra_glm as glm;
use core::panic;
use std::cmp::min_by;
use std::f32::consts::PI;
use std::ptr::null;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};
use rand;

mod shader;
mod util;

use gl::types::GLuint;
use gl::{
    BindBuffer, BindVertexArray, BufferData, GenBuffers, GenVertexArrays, VertexAttribPointer,
    ARRAY_BUFFER, STATIC_DRAW, DrawElements,
};
use glm::pi;
use glutin::event::{
    DeviceEvent,
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;
use rand::Rng;

// initial window size
const INITIAL_SCREEN_W: u32 = 800;
const INITIAL_SCREEN_H: u32 = 600;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //

// Get the size of an arbitrary array of numbers measured in bytes
// Example usage:  pointer_to_array(my_array)
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
// Example usage:  pointer_to_array(my_array)
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
// Example usage:  size_of::<u64>()
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
// Example usage:  offset::<u64>(4)
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()

// == // Generate your VAO here
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>) -> u32 {
    // Implement me!

    // Also, feel free to delete comments :)

    // This should:
    // * Generate a VAO and bind it
    // * Generate a VBO and bind it
    // * Fill it with data
    // * Configure a VAP for the data and enable it
    // * Generate a IBO and bind it
    // * Fill it with data
    // * Return the ID of the VAO

    let mut vao: u32 = 0;

    // We generate and bind (open) the VAO, in order to link the VBOs later
    gl::GenVertexArrays(1, &mut vao);
    assert_ne!(vao, 0);
    gl::BindVertexArray(vao);

    // Now we generate a VBO  to hold the vertex data and we bind it
    let mut vbo: u32 = 0;
    gl::GenBuffers(1, &mut vbo);
    assert_ne!(vbo, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

    // And we fill it with the vertex data
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(&vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW,
    ); // STATIC_DRAW mode for now, we won't make modifications to the buffer

    // Now we create the VAP for the vertex shader data
    // The buffer only contains vertex data, so stride and pointer offset are both
    // 0. We pass an index the 0
    gl::VertexAttribPointer(
        0, 
        3, 
        gl::FLOAT, 
        gl::FALSE, 
        3 * size_of::<f32>(), 
        offset::<f32>(0)
    );

    // And we enable that VAP
    gl::EnableVertexAttribArray(0);

    // Now we start working at the Index Buffer, first we generate it and bind it as ELEMENT_ARRAY_BUFFER
    let mut index_buf: u32 = 0;
    gl::GenBuffers(1, &mut index_buf as *mut u32);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buf);

    // And we fill it with the index data
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW,
    );
    // And we return the VAO id
    return vao;
}

unsafe fn create_circle_vao(cx: f32, cy: f32, r: f32, triangle_count: u32) -> u32{
    /*
    Extra function that creates the VAO for a circle object.
    Parameters:
    cx, cy: The coordinates of the center of the circle
    r: The radius of the circle
    triangle_count: The number of triangles the circle will be approximated from. This function actually creates
    a triangle_count-sided regular polyhedron 
    */

    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
     // Push center vertex
    vertices.push(cx);
    vertices.push(cy);
    vertices.push(0.0);
    let mut theta: f32 = 0.0;
    for x in 1..triangle_count+1 {
        // We build the triangle 0 -> x -> x+1
        // First triangle is center (0) -> 1 -> 2
        // Second triangle is 0 -> 2 -> 3
        // Third triangle is 0 -> 3 -> 4 ...
        // Push new indices
        indices.push(0);
        indices.push(x);
        indices.push(x + 1);
        // Push new edge vertex
        vertices.push(cx + r * theta.cos());
        vertices.push(cy + r * theta.sin());
        vertices.push(0.0);
        // Update angle, we do += pi*2 / triangle_count*2
        theta += 2.0*PI / triangle_count as f32;
        println!("theta = {theta}");
    }
    // Replace last index with 1
    indices.remove(indices.len()-1);
    indices.push(1);
    return create_vao(&vertices, &indices);
}

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(
            INITIAL_SCREEN_W,
            INITIAL_SCREEN_H,
        ));
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!(
                "{}: {}",
                util::get_gl_string(gl::VENDOR),
                util::get_gl_string(gl::RENDERER)
            );
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!(
                "GLSL\t: {}",
                util::get_gl_string(gl::SHADING_LANGUAGE_VERSION)
            );
        }

        // == // Set up your VAO around here
        
        let mut triangle_count: i32= 5;
        
        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0,
            0.86, 0.5, 0.0,
            0.5, 0.86, 0.0,
            -0.5, 0.86, 0.0,
            -0.86, 0.5, 0.0,
            -0.25, -0.96, 0.0,
            0.25, -0.96, 0.0,
            0.12, -0.48, 0.0,
            0.43, 0.25, 0.0,
            -0.43, 0.25, 0.0,
            -0.12, -0.48, 0.0
        ];     
   
        let indices: Vec<u32> = vec![
           0, 1, 2,
           0, 3, 4,
           0, 5, 6,
           0, 7, 8,
           0, 9, 10
        ];   
        let my_vao = unsafe { create_vao(&vertices, &indices) };
        
        // Bonus Challengue - Draw Circle
        /*
        triangle_count = 100;
        let my_vao = unsafe { create_circle_vao(0.0, 0.0, 0.5, triangle_count as u32) };
         */
        
        // == // Set up your shaders here
        let simple_shader;
        unsafe{
            simple_shader = shader::ShaderBuilder::new()
            .attach_file("./shaders/simple.vert")
            .attach_file("./shaders/simple.frag")
            .link();
            simple_shader.activate();
        }
        // Bonus Challengue - Slowly change color
        /*
        let uni_color;
        let start_color = (0.16, 0.72, 0.4);
        let mut frame_color = start_color;
        let mut rng = rand::thread_rng();
        unsafe{
            uni_color = simple_shader.get_uniform_location("uniColor");
        }
        */
        // Basic usage of shader helper:
        // The example code below creates a 'shader' object.
        // It which contains the field `.program_id` and the method `.activate()`.
        // The `.` in the path is relative to `Cargo.toml`.
        // This snippet is not enough to do the exercise, and will need to be modified (outside
        // of just using the correct path), but it only needs to be called once

        // Used to demonstrate keyboard handling for exercise 2.
        let mut _arbitrary_number = 0.0; // feel free to remove

        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut prevous_frame_time = first_frame_time;
        loop {
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(prevous_frame_time).as_secs_f32();
            prevous_frame_time = now;

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Resized");
                    unsafe {
                        gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32);
                    }
                }
            }

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html
                        VirtualKeyCode::A => {
                            _arbitrary_number += delta_time;
                        }
                        VirtualKeyCode::D => {
                            _arbitrary_number -= delta_time;
                        }

                        // default handler:
                        _ => {}
                    }
                }
            }

            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                // == // Optionally access the acumulated mouse movement between
                // == // frames here with `delta.0` and `delta.1`

                *delta = (0.0, 0.0); // reset when done
            }

            // == // Please compute camera transforms here (exercise 2 & 3)

            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky, full opacity
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // == // Issue the necessary gl:: commands to draw your scene here
                // Bonus challengue: Slowly change colour
                /*
                let r_rand =  rng.gen_range(-5..5);
                let g_rand =  rng.gen_range(-5..5);
                let b_rand =  rng.gen_range(-5..5);
                frame_color.0 += r_rand as f32 / 255.0;
                frame_color.1 += g_rand as f32 / 255.0;
                frame_color.2 += b_rand as f32 / 255.0;
                if frame_color.0 < 0.0 {frame_color.0 = 0.0;}
                if frame_color.1 < 0.0 {frame_color.1 = 0.0;}
                if frame_color.1 < 0.0 {frame_color.2 = 0.0;}
                if frame_color.0 > 1.0 {frame_color.0 = 1.0;}
                if frame_color.1 > 1.0 {frame_color.1 = 1.0;}
                if frame_color.1 > 1.0 {frame_color.2 = 1.0;}
                gl::Uniform4f(uni_color, 
                     frame_color.0,  frame_color.1,  frame_color.2,  1.0);
                */
                BindVertexArray(my_vao);
                DrawElements(
                    gl::TRIANGLES, 
                    triangle_count * 3, 
                    gl::UNSIGNED_INT, 
                    offset::<u32>(0)
                );
                
            }

            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
        }
    });

    // == //
    // == // From here on down there are only internals.
    // == //

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(physical_size),
                ..
            } => {
                println!(
                    "New window size! width: {}, height: {}",
                    physical_size.width, physical_size.height
                );
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: key_state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        }
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => {
                        *control_flow = ControlFlow::Exit;
                    }
                    Q => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => {}
        }
    });
}
