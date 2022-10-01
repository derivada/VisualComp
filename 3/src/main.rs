// Uncomment these following global attributes to silence most warnings of "low" interest:

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]

extern crate nalgebra_glm as glm;

mod shader;
mod util;
mod mesh;
mod scene_graph;

use gl::{self, TRIANGLES};
use glm :: {identity, Vec4};
use glutin::event::{
    DeviceEvent,
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;
use shader::Shader;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};
use scene_graph::SceneNode;

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

unsafe fn create_vao_from_mesh(mesh: &mesh::Mesh) -> u32{
    return create_vao(&mesh.vertices, &mesh.indices, &mesh.colors, &mesh.normals);
}

unsafe fn create_vao(
    vertices: &Vec<f32>,
    indices: &Vec<u32>,
    colors: &Vec<f32>,
    normals: &Vec<f32>,
) -> u32 {
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

    // And we also add normal vector data to channel 2
    let mut normal_vbo: u32 = 0;
    gl::GenBuffers(1, &mut normal_vbo);
    assert_ne!(normal_vbo, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, normal_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(&normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(
        2,
        3,
        gl::FLOAT,
        gl::TRUE,
        3 * size_of::<f32>(),
        offset::<f32>(0),
    );
    gl::EnableVertexAttribArray(2);

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

unsafe fn draw_scene(node: &scene_graph::SceneNode,
    view_projection_matrix: &glm::Mat4,
    transformation_so_far: &glm::Mat4,
    vertex_shader: &Shader) {
    // Check if node is drawable, if so: set uniforms and draw
    if node.index_count > 0 {
        gl::BindVertexArray(node.vao_id);
        gl::UniformMatrix4fv(vertex_shader.get_uniform_location("view_proj"), 1, gl::FALSE, view_projection_matrix.as_ptr());
        gl::UniformMatrix4fv(vertex_shader.get_uniform_location("trasnf"), 1, gl::FALSE, transformation_so_far.as_ptr());
        gl::DrawElements(gl::TRIANGLES, node.index_count, gl::UNSIGNED_INT, offset::<u32>(0));
    }
    // Recurse
    for &child in &node.children {
        draw_scene(&*child, view_projection_matrix, transformation_so_far, vertex_shader);
    }   
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

        // Load models and build VAOs
        let moon_mesh: mesh::Mesh = mesh::Terrain::load("./resources/lunarsurface.obj");
        let moon_vao = unsafe {create_vao_from_mesh(&moon_mesh)};
        let helicopter_mesh: mesh::Helicopter = mesh::Helicopter::load("./resources/helicopter.obj");
        let helicopter_body_mesh: mesh::Mesh = helicopter_mesh.body;
        let helicopter_door_mesh: mesh::Mesh = helicopter_mesh.door;
        let helicopter_main_rotor_mesh: mesh::Mesh = helicopter_mesh.main_rotor;
        let helicopter_tail_rotor_mesh: mesh::Mesh = helicopter_mesh.tail_rotor;
        let helicopter_body_vao = unsafe{create_vao_from_mesh(&helicopter_body_mesh)};
        let helicopter_door_vao = unsafe{create_vao_from_mesh(&helicopter_door_mesh)};
        let helicopter_main_rotor_vao = unsafe{create_vao_from_mesh(&helicopter_main_rotor_mesh)};
        let helicopter_tail_rotor_vao = unsafe{create_vao_from_mesh(&helicopter_tail_rotor_mesh)};
                
        // Setup Scene Graph
        let mut scene = SceneNode::new();
        let mut terrain = SceneNode::from_vao(moon_vao, moon_mesh.index_count);
        let mut helicopter_body = SceneNode::from_vao(helicopter_body_vao, helicopter_body_mesh.index_count);
        let mut helicopter_door = SceneNode::from_vao(helicopter_door_vao, helicopter_door_mesh.index_count);
        let mut helicopter_main_rotor = SceneNode::from_vao(helicopter_main_rotor_vao, helicopter_main_rotor_mesh.index_count);
        let mut helicopter_tail_rotor = SceneNode::from_vao(helicopter_tail_rotor_vao, helicopter_tail_rotor_mesh.index_count);
        scene.add_child(&terrain);
        terrain.add_child(&helicopter_body);
        helicopter_body.add_child(&helicopter_door);
        helicopter_body.add_child(&helicopter_main_rotor);
        helicopter_body.add_child(&helicopter_tail_rotor);


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
            let speed_multiplier = 25.0;
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html
                        VirtualKeyCode::W => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(0.0, 0.0, delta_time * speed_multiplier, 1.0);
                            camera_pos += cameraDelta.xyz();
                            // Debug position
                            //println!("Forward z with delta = {}, last rotation matrix = {} camera moved = {}", delta_time, camera_mov_correction, cameraDelta);
                            //println!("New position: {}", camera_pos);
                        }
                        VirtualKeyCode::S => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(0.0, 0.0, -delta_time * speed_multiplier, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::A => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(delta_time * speed_multiplier, 0.0, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::D => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(-delta_time * speed_multiplier, 0.0, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::Space => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(0.0, -delta_time * speed_multiplier, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::LShift => {
                            let cameraDelta: Vec4 = camera_mov_correction
                                * Vec4::new(0.0, delta_time * speed_multiplier, 0.0, 1.0);
                            camera_pos += cameraDelta.xyz();
                        }
                        VirtualKeyCode::Up => {
                            // Here we limit camera pitch movement so we can only rotate it to straight up or straight down
                            camera_pitch = (camera_pitch - delta_time).min(PI / 2.0);
                        }
                        VirtualKeyCode::Down => {
                            camera_pitch = (camera_pitch + delta_time).max(-PI / 2.0);
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
            // println!("x = {}, y = {}, z = {}, yaw = {}, pitch = {}", camera_pos.x, camera_pos.y, camera_pos.z, camera_yaw, camera_pitch);
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                // == // Optionally access the acumulated mouse movement between
                // == // frames here with `delta.0` and `delta.1`

                *delta = (0.0, 0.0); // reset when done
            }
            // Camera transforms computing
            // Perspective transformation, we choose FOV = 60º
            let persp: glm::Mat4 = glm::perspective(window_aspect_ratio, PI/3.0, 1.0, 1000.0);
            // xyz camera position translation
            let camera_transl: glm::Mat4 = glm::translation(&camera_pos);
            // Yaw rotation
            let rot_yaw_axis: glm::Vec3 = glm::Vec3::new(0.0, 1.0, 0.0);
            let rot_yaw: glm::Mat4 = glm::rotation( camera_yaw, &rot_yaw_axis);
            // Pitch rotation
            let rot_pitch_axis: glm::Vec3 = glm::Vec3::new(1.0, 0.0, 0.0);
            let rot_pitch: glm::Mat4 = glm::rotation( camera_pitch, &rot_pitch_axis);

            let camera_rot: glm::Mat4 = rot_pitch * rot_yaw;

            // Inverse rotation matrix (for the voluntary ex. 1 camera movement correction, view key handler for use)
            // We don't invert the rotation matrix, we create another one with a inverse (-deegree) rotation, which is faster
            camera_mov_correction = glm::rotation(-camera_yaw, &rot_yaw_axis)
                * glm::rotation(-camera_pitch, &rot_pitch_axis);

            // Compute the final transformation passed to the vertex shader
            let transf: glm::Mat4 = persp * camera_rot * camera_transl;
            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky, full opacity
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Scene drawing
                let id: glm::Mat4 = glm::identity();
                draw_scene(&scene, &transf, &id, &simple_shader);
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
