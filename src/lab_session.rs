//! Each scene owns a renderer worker because upstream kernel memoization can
//! retain GPU resource bindings until process exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    Gallery,
    WhiteRoom,
    Courtyard,
    Sponza,
    IntelSponza,
}

impl Scene {
    pub const ALL: [Self; 5] = [
        Self::Gallery,
        Self::WhiteRoom,
        Self::Courtyard,
        Self::Sponza,
        Self::IntelSponza,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Gallery => "Chromatic Gallery",
            Self::WhiteRoom => "White light-leak lab",
            Self::Courtyard => "Small courtyard",
            Self::Sponza => "Sponza OBJ",
            Self::IntelSponza => "Intel Sponza",
        }
    }
    pub fn flag(self) -> &'static str {
        match self {
            Self::Gallery => "--gallery",
            Self::WhiteRoom => "--white-room",
            Self::Courtyard => "--courtyard",
            Self::Sponza => "--sponza",
            Self::IntelSponza => "--intel-sponza",
        }
    }
    pub fn exit_code(self) -> i32 {
        70 + Self::ALL.iter().position(|v| *v == self).unwrap() as i32
    }
    fn from_exit(code: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.exit_code() == code)
    }
}

fn switched_arguments(arguments: &[String], scene: Scene) -> Vec<String> {
    let mut result = Vec::new();
    let mut skip_value = false;
    for arg in arguments {
        if skip_value {
            skip_value = false;
            continue;
        }
        if ["--view", "--scene-path", "--cubes", "--emission"].contains(&arg.as_str()) {
            skip_value = true;
            continue;
        }
        if Scene::ALL.iter().any(|s| s.flag() == arg)
            || ["--scene-worker", "--emissive-day", "--emissive-night"].contains(&arg.as_str())
        {
            continue;
        }
        result.push(arg.clone());
    }
    result.push(scene.flag().into());
    // Preserve the externally supplied Intel asset when returning to its scene.
    if scene == Scene::IntelSponza {
        if let Some(i) = arguments.iter().position(|s| s == "--scene-path") {
            if let Some(path) = arguments.get(i + 1) {
                result.extend(["--scene-path".into(), path.clone()]);
            }
        }
    }
    result
}

/// Returns only inside a worker (or for --help). The supervisor creates no GPU
/// context and waits for the old worker to exit before starting the next one.
pub fn supervise() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|s| s == "--scene-worker" || s == "--help")
    {
        return;
    }
    let mut worker_arguments = arguments.clone();
    loop {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(&worker_arguments)
            .arg("--scene-worker")
            .status()
            .expect("Could not start scene worker");
        match status.code().and_then(Scene::from_exit) {
            Some(scene) => {
                worker_arguments = switched_arguments(&arguments, scene);
            }
            None => std::process::exit(status.code().unwrap_or(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn switch_preserves_render_options_and_replaces_scene_arguments() {
        let args = [
            "--gallery",
            "--1080p",
            "--view",
            "2",
            "--probe-budget",
            "1024",
            "--emissive-night",
            "--emission",
            "4",
            "--scene-path",
            "my asset/scene.gltf",
        ]
        .map(str::to_string);
        for scene in Scene::ALL {
            assert_eq!(Scene::from_exit(scene.exit_code()), Some(scene));
            let next = switched_arguments(&args, scene);
            assert!(next.contains(&"--1080p".into()));
            assert!(next.windows(2).any(|v| v == ["--probe-budget", "1024"]));
            assert_eq!(
                next.iter()
                    .filter(|a| Scene::ALL.iter().any(|s| s.flag() == a.as_str()))
                    .count(),
                1
            );
            assert!(!next.contains(&"--view".into()));
            assert!(!next.contains(&"--emissive-night".into()));
            assert_eq!(
                next.contains(&"--scene-path".into()),
                scene == Scene::IntelSponza
            );
        }
    }
}
