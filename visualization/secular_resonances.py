from manim import *
import numpy as np


class SecularResonance(Scene):
    def construct(self):
        master_scene_group = Group()

        # Configuration
        sun_img_file = "sun.png"
        mercury_img_file = "mercury.png"
        jupiter_img_file = "jupiter.png"

        sun_scale = 0.1
        mercury_a = 1.0
        mercury_e = 0.4
        mercury_orbit_color = BLUE
        mercury_scale = 0.03
        mercury_orbit_period = 8

        jupiter_a = 2.75
        jupiter_e = 0.4
        jupiter_orbit_color = YELLOW
        jupiter_scale = 0.04
        jupiter_orbit_period = 20

        mercury_precession_rate_non_resonant = 0.2
        jupiter_precession_rate_non_resonant = 0.35
        resonant_precession_rate = 0.25
        animation_phase_duration = 15

        # Helper function to create an orbit system
        def create_celestial_body_system(semi_major_axis, eccentricity, orbit_color, planet_img_file, planet_scale, initial_precession_angle=0, sun_at_left_focus=False):
            focus_dist = semi_major_axis * eccentricity
            semi_minor_axis = semi_major_axis * np.sqrt(1 - eccentricity**2)
            if semi_minor_axis <= 0:
                semi_minor_axis = 0.01  # Prevent zero/negative height for flat ellipses

            shift_direction = RIGHT if sun_at_left_focus else LEFT
            orbit_ellipse = Ellipse(width=2 * semi_major_axis, height=2 * semi_minor_axis, color=orbit_color).shift(focus_dist * shift_direction)  # Shift so one focus is at ORIGIN

            planet = ImageMobject(planet_img_file).scale(planet_scale)
            orbit_ellipse.rotate(initial_precession_angle, about_point=ORIGIN)
            return orbit_ellipse, planet

        # Scene Setup
        sun = ImageMobject(sun_img_file).scale(sun_scale).move_to(ORIGIN)

        mercury_orbit_ellipse, mercury_planet_mobj = create_celestial_body_system(mercury_a, mercury_e, mercury_orbit_color, mercury_img_file, mercury_scale)
        jupiter_orbit_ellipse, jupiter_planet_mobj = create_celestial_body_system(jupiter_a, jupiter_e, jupiter_orbit_color, jupiter_img_file, jupiter_scale, initial_precession_angle=PI / 4, sun_at_left_focus=True)

        master_scene_group.add(mercury_orbit_ellipse, jupiter_orbit_ellipse)
        master_scene_group.add(mercury_planet_mobj, jupiter_planet_mobj)
        self.add(master_scene_group)

        self.add(sun)
        self.bring_to_front(sun)

        mercury_precession_angle = ValueTracker(0)
        jupiter_precession_angle = ValueTracker(PI / 4)
        mercury_orbital_angle = ValueTracker(0)
        jupiter_orbital_angle = ValueTracker(0)

        def mercury_orbit_updater(mobj, dt):
            current_time = self.renderer.time
            rate = resonant_precession_rate if current_time >= animation_phase_duration else mercury_precession_rate_non_resonant
            mercury_precession_angle.increment_value(rate * dt)
            new_semi_minor = mercury_a * np.sqrt(1 - mercury_e**2)
            if new_semi_minor <= 0:
                new_semi_minor = 0.01
            mobj.become(Ellipse(width=2 * mercury_a, height=2 * new_semi_minor, color=mercury_orbit_color).shift(mercury_a * mercury_e * LEFT).rotate(mercury_precession_angle.get_value(), about_point=ORIGIN))

        def jupiter_orbit_updater(mobj, dt):
            current_time = self.renderer.time
            rate = resonant_precession_rate if current_time >= animation_phase_duration else jupiter_precession_rate_non_resonant
            jupiter_precession_angle.increment_value(rate * dt)
            new_semi_minor = jupiter_a * np.sqrt(1 - jupiter_e**2)
            if new_semi_minor <= 0:
                new_semi_minor = 0.01
            mobj.become(Ellipse(width=2 * jupiter_a, height=2 * new_semi_minor, color=jupiter_orbit_color).shift(jupiter_a * jupiter_e * RIGHT).rotate(jupiter_precession_angle.get_value(), about_point=ORIGIN))

        mercury_orbit_ellipse.add_updater(mercury_orbit_updater)
        jupiter_orbit_ellipse.add_updater(jupiter_orbit_updater)

        def update_mercury_pos(mob):
            alpha = (mercury_orbital_angle.get_value() / (2 * PI)) % 1.0
            mob.move_to(mercury_orbit_ellipse.point_from_proportion(alpha))

        def update_jupiter_pos(mob):
            alpha = (jupiter_orbital_angle.get_value() / (2 * PI)) % 1.0
            mob.move_to(jupiter_orbit_ellipse.point_from_proportion(alpha))

        mercury_planet_mobj.add_updater(update_mercury_pos)
        jupiter_planet_mobj.add_updater(update_jupiter_pos)

        # Animation Phases
        # Phase 1: Non-resonant
        status_text_phase1 = Text("Nicht in säkularer Resonanz", font_size=24, color=WHITE).to_corner(UL)
        self.play(Write(status_text_phase1))

        mercury_revolutions_phase1 = animation_phase_duration / mercury_orbit_period
        jupiter_revolutions_phase1 = animation_phase_duration / jupiter_orbit_period
        target_mercury_angle_phase1 = mercury_revolutions_phase1 * 2 * PI
        target_jupiter_angle_phase1 = jupiter_revolutions_phase1 * 2 * PI

        self.play(mercury_orbital_angle.animate.set_value(target_mercury_angle_phase1), jupiter_orbital_angle.animate.set_value(target_jupiter_angle_phase1), run_time=animation_phase_duration, rate_func=linear)

        # Phase 2: Resonant
        status_text_phase2 = Text("In säkularer Resonanz", font_size=24, color=WHITE).to_corner(UL)
        self.play(Transform(status_text_phase1, status_text_phase2))
        # status_text_phase1 is now transformed into status_text_phase2, so we can refer to it as status_text_phase2 if needed later.

        total_duration = 2 * animation_phase_duration  # Total duration of orbital animations
        mercury_revolutions_total = total_duration / mercury_orbit_period
        jupiter_revolutions_total = total_duration / jupiter_orbit_period
        target_mercury_angle_total = mercury_revolutions_total * 2 * PI
        target_jupiter_angle_total = jupiter_revolutions_total * 2 * PI

        self.play(mercury_orbital_angle.animate.set_value(target_mercury_angle_total), jupiter_orbital_angle.animate.set_value(target_jupiter_angle_total), run_time=animation_phase_duration, rate_func=linear)

        # Cleanup updaters
        mercury_orbit_ellipse.remove_updater(mercury_orbit_updater)
        jupiter_orbit_ellipse.remove_updater(jupiter_orbit_updater)
        mercury_planet_mobj.remove_updater(update_mercury_pos)
        jupiter_planet_mobj.remove_updater(update_jupiter_pos)
