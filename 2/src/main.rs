// Uncomment these following global attributes to silence most warnings of "low" interest:

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]

extern crate nalgebra_glm as glm;
use core::panic;
use rand;
use std::cmp::min_by;
use std::f32::consts::PI;
use std::ptr::null;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};

mod shader;
mod util;

use gl::types::GLuint;
use gl::{
    BindBuffer, BindVertexArray, BufferData, DrawElements, GenBuffers, GenVertexArrays,
    VertexAttribPointer, ARRAY_BUFFER, STATIC_DRAW,
};
use glm::{diagonal4x4, identity, pi, vec4, Vec4};
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
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>) -> u32 {
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

    // Now we generate a VBO to hold the vertex data and we bind it
    let mut vertex_vbo: u32 = 0;
    gl::GenBuffers(1, &mut vertex_vbo);
    assert_ne!(vertex_vbo, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, vertex_vbo);

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
        offset::<f32>(0),
    );

    // And we enable that VAP
    gl::EnableVertexAttribArray(0);

    // Now we have to add the color data, we create another buffer and attribute pointer
    let mut color_vbo: u32 = 0;
    gl::GenBuffers(1, &mut color_vbo);
    assert_ne!(color_vbo, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(&colors),
        pointer_to_array(colors),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(
        1,
        4,
        gl::FLOAT,
        gl::FALSE,
        4 * size_of::<f32>(),
        offset::<f32>(0),
    );
    gl::EnableVertexAttribArray(1);

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

fn add_circle(
    cx: f32,
    cy: f32,
    z: f32,
    r: f32,
    triangle_count: u32,
    vertices: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) -> u32 {
    // Push center vertex
    let mut newIndices: Vec<u32> = Vec::new();
    vertices.push(cx);
    vertices.push(cy);
    vertices.push(0.0);
    let mut theta: f32 = 0.0;
    for x in 1..triangle_count + 1 {
        newIndices.push(0);
        newIndices.push(x as u32);
        newIndices.push((x + 1) as u32);
        vertices.push(cx + r * theta.cos());
        vertices.push(cy + r * theta.sin());
        vertices.push(z);
        theta += 2.0 * PI / triangle_count as f32;
        println!("theta = {theta}");
    }
    newIndices.remove(indices.len() - 1);
    newIndices.push(1);
    let mut max_value = 0;
    if indices.len() > 0 {
        max_value = indices.iter().copied().max().unwrap() + 1;
    }
    newIndices = newIndices.iter().map(|x| x + max_value).collect();
    indices.append(&mut newIndices);
    return triangle_count;
}

fn add_equilateral_triangle(
    cx: f32,
    cy: f32,
    z: f32,
    dir: f32,
    radius: f32,
    vertices: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) -> u32 {
    let mut theta = dir;
    for i in 0..3 {
        vertices.push(cx + theta.cos() * radius);
        vertices.push(cy + theta.sin() * radius);
        vertices.push(z);
        theta += 2.0 * PI / 3.0;
    }
    let mut newIndices = vec![0, 1, 2];
    let mut max_value = 0;
    if indices.len() > 0 {
        max_value = indices.iter().copied().max().unwrap() + 1;
    }
    newIndices = newIndices.iter().map(|x| x + max_value).collect();
    indices.append(&mut newIndices);
    return 1;
}

fn add_rectangle(
    cx: f32,
    cy: f32,
    z: f32,
    width: f32,
    height: f32,
    vertices: &mut Vec<f32>,
    indices: &mut Vec<u32>,
) -> u32 {
    let hw = width / 2.0;
    let hh = height / 2.0;
    let mut newVertices: Vec<f32> = vec![
        cx - hw,
        cy + hh,
        z,
        cx - hw,
        cy - hh,
        z,
        cx + hw,
        cy - hh,
        z,
        cx + hw,
        cy + hh,
        z,
    ];
    vertices.append(&mut newVertices);
    let mut newIndices = vec![0, 1, 2, 0, 2, 3];
    let mut max_value = 0;
    if indices.len() > 0 {
        max_value = indices.iter().copied().max().unwrap() + 1;
    }
    newIndices = newIndices.iter().map(|x| x + max_value).collect();
    indices.append(&mut newIndices);
    return 2;
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
        let mut vertices: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut colors: Vec<f32> = Vec::new();
        let mut triangle_count: u32 = 0;
        
        // Task 1 composition
        /*
        // Triangles with interpolated colouring
        let mut colors1: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let mut colors2: Vec<f32> = vec![ 0.27, 0.66, 0.83, 1.0, 0.34, 0.12, 0.92, 1.0, 0.07, 0.73, 0.62, 1.0];
        let mut colors3: Vec<f32> = vec![0.9, 0.198, 0.72, 1.0, 0.72, 0.07, 0.37, 1.0, 0.43, 0.18, 0.63, 1.0];
        triangle_count += add_equilateral_triangle(0.7, 0.7, 0.0, PI / 2.0, 0.2, &mut vertices, &mut indices);
        colors.append(&mut colors1);
        triangle_count += add_equilateral_triangle(-0.7, -0.7, 0.0, PI, 0.3, &mut vertices, &mut indices);
        colors.append(&mut colors2);
        triangle_count += add_equilateral_triangle(
            -0.7,
            0.7,
            0.0,
            3.0 * PI / 2.0,
            0.1,
            &mut vertices,
            &mut indices,
        );
        colors.append(&mut colors3);

        // Spanish flag
        let mut up: Vec<f32> = Vec::new();
        let mut mid: Vec<f32> = Vec::new();
        let mut low: Vec<f32> = Vec::new();
        let red = [1.0, 0.0, 0.0, 1.0];
        let yellow = [1.0, 1.0, 0.0, 1.0];
        for i in 0..4 {
            up.extend_from_slice(&red);
            mid.extend_from_slice(&yellow);
            low.extend_from_slice(&red);
        }
        triangle_count += add_rectangle(0.0, -0.3, 0.0, 1.2, 0.2, &mut vertices, &mut indices);
        colors.append(&mut up);
        triangle_count += add_rectangle(0.0, -0.0, 0.0, 1.2, 0.4, &mut vertices, &mut indices);
        colors.append(&mut mid);
        triangle_count += add_rectangle(0.0, 0.3, 0.0, 1.2, 0.2, &mut vertices, &mut indices);
        colors.append(&mut low);
         */
        // Task 2 composition
        
        // Basic RGB triangles in different angles, sizes and depth values
        // All alpha values are 0.5
        let mut red: Vec<f32> = vec![1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5];
        let mut green: Vec<f32> = vec![0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5];
        let mut blue: Vec<f32> = vec![0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5];
        triangle_count += add_equilateral_triangle(0.0, 0.15, -10.0, PI / 2.0, 0.7, &mut vertices, &mut indices); // Far 
        colors.append(&mut red);
        triangle_count += add_equilateral_triangle(-0.05, -0.05,-5.0, 0.3, 0.45, &mut vertices, &mut indices); // Middle
        colors.append(&mut green);
        triangle_count += add_equilateral_triangle(0.1, 0.1, 0.0, 3.0 * PI / 2.0,0.35, &mut vertices, &mut indices); // Near
        colors.append(&mut blue);

        // Build the VAO with the previous specified geometry
        let my_vao = unsafe { create_vao(&vertices, &indices, &colors) };

        // Shader setup
        let simple_shader;
        unsafe {
            simple_shader = shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link();
            simple_shader.activate();
        }

        // Initial camera position
        let mut camera_pos: glm::Vec3 = glm::Vec3::new(0.0, 0.0, 0.0);
        let mut camera_yaw: f32 = 0.0;
        let mut camera_pitch: f32 = 0.0;
        let mut camera_mov_correction: glm::Mat4 = identity();

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
                        VirtualKeyCode::W => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(0.0, 0.0, delta_time, 1.0);
                            camera_pos += cameraDelta.xyz();
                            // Debug position
                            //println!("Forward z with delta = {}, last rotation matrix = {} camera moved = {}", delta_time, camera_mov_correction, cameraDelta);
                            //println!("New position: {}", camera_pos);
                        }
                        VirtualKeyCode::S => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(0.0, 0.0, -delta_time, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::A => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(-delta_time, 0.0, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::D => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(delta_time, 0.0, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::Space => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(0.0, delta_time, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::LShift => {
                            let cameraDelta: Vec4 =
                                camera_mov_correction * Vec4::new(0.0, -delta_time, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::Up => {
                            // Here we limit camera pitch movement so we can only rotate it to straight up or straight down
                            camera_pitch = (camera_pitch + delta_time).min(PI / 2.0);
                        }
                        VirtualKeyCode::Down => {
                            camera_pitch = (camera_pitch - delta_time).max(-PI / 2.0);
                        }
                        VirtualKeyCode::Left => {
                            camera_yaw -= delta_time;
                        }
                        VirtualKeyCode::Right => {
                            camera_yaw += delta_time;
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
            /*
            // Task 3b) transformations
            let mut transf: glm::Mat4 = glm::identity();
            //transf[(0, 0)] = elapsed.sin(); // X - scaling
            //transf[(0, 1)] = elapsed.sin(); // X - shearing
            //transf[(0, 3)] = elapsed.sin(); // X - translating
            //transf[(1, 0)] = elapsed.sin(); // Y - shearing
            //transf[(1, 1)] = elapsed.sin(); // Y - scaling
            //transf[(1, 3)] = elapsed.sin(); // Y - translating
            */
            
            // Camera transforms computing
            let id: glm::Mat4 = glm::identity();
            // Translation to move the scene contents from [-1, 1] to [-3, -1],
            // so they can be seen after the projection that flips the axis
            let translvec_persp: glm::Vec3 = glm::Vec3::new(0.0, 0.0, -2.0);
            let transl_persp: glm::Mat4 = glm::translate(&id, &translvec_persp);
            // Perspective transformation, we choose FOV = 100
            let persp: glm::Mat4 = glm::perspective(window_aspect_ratio, 100.0, 1.0, 100.0);
            // xyz camera position translation
            let camera_transl: glm::Mat4 = glm::translate(&id, &camera_pos);
            // Yaw rotation
            let rot_yaw_axis: glm::Vec3 = glm::Vec3::new(0.0, 1.0, 0.0);
            let rot_yaw: glm::Mat4 = glm::rotate(&id, -camera_yaw, &rot_yaw_axis);
            // Pitch rotation
            let rot_pitch_axis: glm::Vec3 = glm::Vec3::new(1.0, 0.0, 0.0);
            let rot_pitch: glm::Mat4 = glm::rotate(&id, camera_pitch, &rot_pitch_axis);

            let camera_rot: glm::Mat4 = rot_pitch * rot_yaw;

            // Inverse rotation matrix (for the voluntary ex. 1 camera movement correction, view key handler for use)
            // We don't invert the rotation matrix, we create another one with a inverse (-deegree) rotation, which is faster
            camera_mov_correction = glm::rotate(&id, camera_yaw, &rot_yaw_axis)
                * glm::rotate(&id, -camera_pitch, &rot_pitch_axis);

            // Compute the final transformation passed to the vertex shader
            let transf: glm::Mat4 = persp * transl_persp * camera_rot * camera_transl;
            
            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky, full opacity
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Scene drawing
                BindVertexArray(my_vao);
                // We pass the transformation matrix to the vertex shader as an uniform variable
                gl::UniformMatrix4fv(2, 1, gl::FALSE, transf.as_ptr());
                DrawElements(
                    gl::TRIANGLES,
                    (triangle_count * 3) as i32,
                    gl::UNSIGNED_INT,
                    offset::<u32>(0),
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
