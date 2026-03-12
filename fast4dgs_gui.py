#!/usr/bin/env python3

from __future__ import annotations

import argparse
import threading
import traceback
import webbrowser
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, scrolledtext, ttk

from fast4dgs_app import (
    AppPaths,
    Fast4DGSController,
    RuntimeOptions,
    ViewerOptions,
    resize_for_display,
)


class Fast4DGSGui:
    CONTINUOUS_POINTCLOUD_UPDATE_MS = 1500

    def __init__(self, root: tk.Tk, controller: Fast4DGSController) -> None:
        self.root = root
        self.controller = controller
        self.root.title("Fast4DGS Control Panel")
        self.root.geometry("1800x1100")
        self.root.minsize(1440, 900)

        self.stream_config_path_var = tk.StringVar(value=str(self.controller.paths.stream_config_path))
        self.pp_config_path_var = tk.StringVar(value=str(self.controller.paths.pp_pointcloud_config_path))
        self.colmap_path_var = tk.StringVar(value=str(self.controller.paths.colmap_path))
        self.sam_checkpoint_var = tk.StringVar(value=str(self.controller.paths.sam_checkpoint))
        self.sam_model_cfg_var = tk.StringVar(value=str(self.controller.paths.sam_model_cfg))
        self.skip_seconds_var = tk.StringVar(value=str(self.controller.options.skip_seconds))
        self.startup_timeout_var = tk.StringVar(value=str(self.controller.options.startup_timeout))
        self.point_size_var = tk.StringVar(value=str(self.controller.options.point_size))
        self.marker_size_var = tk.StringVar(value=str(self.controller.options.marker_size))
        self.viser_port_var = tk.StringVar(value="8080")

        self.publish_ros_pointcloud_var = tk.BooleanVar(value=False)
        self.publish_aruco_markers_var = tk.BooleanVar(value=False)
        self.publish_target_position_var = tk.BooleanVar(value=False)
        self.track_target_var = tk.BooleanVar(value=False)

        self.show_aruco_in_viewer_var = tk.BooleanVar(value=True)
        self.show_target_bbox_in_viewer_var = tk.BooleanVar(value=True)
        self.show_cameras_in_viewer_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Load configs, then click Save and Start.")
        self.viewer_status_var = tk.StringVar(value="Viser stopped")
        self.pointcloud_status_var = tk.StringVar(value="Point cloud not computed yet.")

        self.stream_config_text: scrolledtext.ScrolledText | None = None
        self.pp_config_text: scrolledtext.ScrolledText | None = None
        self.target_stream_combo: ttk.Combobox | None = None
        self.individual_stream_combo: ttk.Combobox | None = None

        self.select_canvas: tk.Canvas | None = None
        self.select_canvas_photo = None
        self.select_canvas_scale = 1.0
        self.select_original_frame = None
        self.select_rectangle_id = None
        self.drag_start = None
        self.current_raw_box = None
        self.current_preview_display_name = None

        self.overlay_preview_label: ttk.Label | None = None
        self.mask_overlay_label: ttk.Label | None = None
        self.all_streams_label: ttk.Label | None = None
        self.individual_stream_label: ttk.Label | None = None
        self.selected_overlay_label: ttk.Label | None = None

        self._all_streams_photo = None
        self._individual_stream_photo = None
        self._mask_overlay_photo = None
        self._selected_overlay_photo = None
        self._continuous_viewer_updates_enabled = False
        self._pointcloud_update_in_progress = False
        self._pointcloud_after_id = None

        self._build_ui()
        self._load_config_editors()
        self._update_tab_states(session_ready=False, mask_ready=False)
        self._schedule_stream_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)

        status_frame = ttk.Frame(container)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").grid(row=0, column=0, sticky="ew")

        self.main_notebook = ttk.Notebook(container)
        self.main_notebook.grid(row=1, column=0, sticky="nsew")

        self.config_tab = ttk.Frame(self.main_notebook, padding=10)
        self.select_target_tab = ttk.Frame(self.main_notebook, padding=10)
        self.view_streams_tab = ttk.Frame(self.main_notebook, padding=10)
        self.view_pointcloud_tab = ttk.Frame(self.main_notebook, padding=10)

        self.main_notebook.add(self.config_tab, text="Config")
        self.main_notebook.add(self.select_target_tab, text="Select Target")
        self.main_notebook.add(self.view_streams_tab, text="View Streams")
        self.main_notebook.add(self.view_pointcloud_tab, text="View Point Cloud")

        self._build_config_tab()
        self._build_select_target_tab()
        self._build_view_streams_tab()
        self._build_view_pointcloud_tab()

    def _build_config_tab(self) -> None:
        self.config_tab.rowconfigure(0, weight=1)
        self.config_tab.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self.config_tab)
        notebook.grid(row=0, column=0, sticky="nsew")

        stream_config_frame = ttk.Frame(notebook, padding=10)
        pp_config_frame = ttk.Frame(notebook, padding=10)
        other_options_frame = ttk.Frame(notebook, padding=10)

        notebook.add(stream_config_frame, text="Stream Config")
        notebook.add(pp_config_frame, text="Point-cloud Post-processing")
        notebook.add(other_options_frame, text="Other Options")

        self._build_json_editor(
            stream_config_frame,
            path_var=self.stream_config_path_var,
            title="Edit stream_config.json",
            assign=lambda widget: setattr(self, "stream_config_text", widget),
            save_callback=self.on_save_stream_config,
            reload_callback=self.on_reload_stream_config,
        )
        self._build_json_editor(
            pp_config_frame,
            path_var=self.pp_config_path_var,
            title="Edit pp_pointcloud.json",
            assign=lambda widget: setattr(self, "pp_config_text", widget),
            save_callback=self.on_save_pp_config,
            reload_callback=self.on_reload_pp_config,
        )

        self._build_other_options_tab(other_options_frame)

        button_frame = ttk.Frame(self.config_tab)
        button_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(button_frame, text="Save and Start", command=self.on_save_and_start).pack(side="right")

    def _build_json_editor(self, parent, path_var, title, assign, save_callback, reload_callback) -> None:
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text=title).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="Path:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        ttk.Entry(parent, textvariable=path_var).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        button_row = ttk.Frame(parent)
        button_row.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(0, 8))
        ttk.Button(button_row, text="Reload", command=reload_callback).pack(side="left", padx=(0, 6))
        ttk.Button(button_row, text="Save", command=save_callback).pack(side="left")

        text_widget = scrolledtext.ScrolledText(parent, wrap="none", undo=True, font=("DejaVu Sans Mono", 10))
        text_widget.grid(row=2, column=0, columnspan=3, sticky="nsew")
        assign(text_widget)

    def _build_other_options_tab(self, parent) -> None:
        parent.columnconfigure(1, weight=1)

        publish_frame = ttk.LabelFrame(parent, text="Publishing", padding=10)
        publish_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Checkbutton(publish_frame, text="Publish ROS pointcloud", variable=self.publish_ros_pointcloud_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(publish_frame, text="Publish ArUco markers", variable=self.publish_aruco_markers_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(publish_frame, text="Publish target position", variable=self.publish_target_position_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(publish_frame, text="Track target (for later)", variable=self.track_target_var, state="disabled").grid(row=3, column=0, sticky="w")

        advanced_frame = ttk.LabelFrame(parent, text="Paths and Runtime", padding=10)
        advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        advanced_frame.columnconfigure(1, weight=1)
        advanced_frame.columnconfigure(3, weight=1)

        fields = [
            ("COLMAP path", self.colmap_path_var),
            ("SAM checkpoint", self.sam_checkpoint_var),
            ("SAM model cfg", self.sam_model_cfg_var),
            ("Skip seconds", self.skip_seconds_var),
            ("Startup timeout", self.startup_timeout_var),
            ("Point size", self.point_size_var),
            ("Marker size", self.marker_size_var),
            ("Viser port", self.viser_port_var),
        ]

        for row, (label, variable) in enumerate(fields):
            ttk.Label(advanced_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(advanced_frame, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)

    def _build_select_target_tab(self) -> None:
        self.select_target_tab.rowconfigure(0, weight=1)
        self.select_target_tab.columnconfigure(0, weight=1)

        self.target_notebook = ttk.Notebook(self.select_target_tab)
        self.target_notebook.grid(row=0, column=0, sticky="nsew")

        select_frame = ttk.Frame(self.target_notebook, padding=10)
        overlay_frame = ttk.Frame(self.target_notebook, padding=10)
        self.target_notebook.add(select_frame, text="1. Select Snapshot")
        self.target_notebook.add(overlay_frame, text="2. Review Masks")

        self.target_notebook.tab(1, state="disabled")

        select_frame.rowconfigure(1, weight=1)
        select_frame.columnconfigure(0, weight=2)
        select_frame.columnconfigure(1, weight=1)

        control_frame = ttk.Frame(select_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(control_frame, text="Stream:").pack(side="left")
        self.target_stream_combo = ttk.Combobox(control_frame, state="readonly", width=48)
        self.target_stream_combo.pack(side="left", padx=(8, 8))
        self.target_stream_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_target_preview())
        ttk.Button(control_frame, text="Refresh Snapshot", command=self.on_refresh_target_snapshot).pack(side="left", padx=(0, 8))
        ttk.Button(control_frame, text="Clear Box", command=self.clear_box_selection).pack(side="left")

        canvas_frame = ttk.Frame(select_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.select_canvas = tk.Canvas(canvas_frame, bg="#111111", highlightthickness=1, highlightbackground="#444444")
        self.select_canvas.grid(row=0, column=0, sticky="nsew")
        self.select_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.select_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.select_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        right_panel = ttk.Frame(select_frame)
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        ttk.Label(
            right_panel,
            text="Draw a bounding box on the selected snapshot, then run SAM.",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(right_panel, text="Run SAM", command=self.on_run_sam).grid(row=1, column=0, sticky="ew", pady=(10, 6))
        ttk.Button(right_panel, text="Save Mask to Memory", command=self.on_save_mask_to_memory).grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Button(right_panel, text="Reselect Mask", command=self.on_reselect_mask).grid(row=3, column=0, sticky="ew", pady=6)

        ttk.Label(right_panel, text="Selected stream overlay").grid(row=4, column=0, sticky="w", pady=(12, 6))
        self.selected_overlay_label = ttk.Label(right_panel)
        self.selected_overlay_label.grid(row=5, column=0, sticky="nsew")

        overlay_frame.rowconfigure(0, weight=1)
        overlay_frame.columnconfigure(0, weight=1)
        ttk.Label(overlay_frame, text="Masks overlaid on the latest snapshots from all streams").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.mask_overlay_label = ttk.Label(overlay_frame)
        self.mask_overlay_label.grid(row=1, column=0, sticky="nsew")

    def _build_view_streams_tab(self) -> None:
        self.view_streams_tab.rowconfigure(0, weight=1)
        self.view_streams_tab.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self.view_streams_tab)
        notebook.grid(row=0, column=0, sticky="nsew")

        all_streams_frame = ttk.Frame(notebook, padding=10)
        single_stream_frame = ttk.Frame(notebook, padding=10)
        notebook.add(all_streams_frame, text="View All Streams")
        notebook.add(single_stream_frame, text="View Individual Stream")

        all_streams_frame.rowconfigure(0, weight=1)
        all_streams_frame.columnconfigure(0, weight=1)
        self.all_streams_label = ttk.Label(all_streams_frame)
        self.all_streams_label.grid(row=0, column=0, sticky="nsew")

        single_stream_frame.rowconfigure(1, weight=1)
        single_stream_frame.columnconfigure(0, weight=1)
        controls = ttk.Frame(single_stream_frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Stream:").pack(side="left")
        self.individual_stream_combo = ttk.Combobox(controls, state="readonly", width=48)
        self.individual_stream_combo.pack(side="left", padx=(8, 0))
        self.individual_stream_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_individual_stream_view())
        self.individual_stream_label = ttk.Label(single_stream_frame)
        self.individual_stream_label.grid(row=1, column=0, sticky="nsew")

    def _build_view_pointcloud_tab(self) -> None:
        self.view_pointcloud_tab.columnconfigure(0, weight=1)

        top = ttk.Frame(self.view_pointcloud_tab)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Viser status:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.viewer_status_var).grid(row=0, column=1, sticky="w", padx=(8, 0))

        button_row = ttk.Frame(self.view_pointcloud_tab)
        button_row.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        ttk.Button(button_row, text="Open Viser", command=self.on_open_viser).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Update Point Cloud", command=self.on_update_point_cloud).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Stop Viser", command=self.on_stop_viser).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Open in Browser", command=self.on_open_viewer_in_browser).pack(side="left")

        options = ttk.LabelFrame(self.view_pointcloud_tab, text="Viewer Options", padding=10)
        options.grid(row=2, column=0, sticky="ew")
        ttk.Checkbutton(options, text="Enable/disable ArUco markers", variable=self.show_aruco_in_viewer_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Show target bound box", variable=self.show_target_bbox_in_viewer_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options, text="Show camera position", variable=self.show_cameras_in_viewer_var).grid(row=2, column=0, sticky="w")

        status_frame = ttk.LabelFrame(self.view_pointcloud_tab, text="Status", padding=10)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.pointcloud_status_var, wraplength=1200, justify="left").grid(row=0, column=0, sticky="w")

    def _update_tab_states(self, session_ready: bool, mask_ready: bool) -> None:
        self.main_notebook.tab(1, state="normal" if session_ready else "disabled")
        self.main_notebook.tab(2, state="normal" if session_ready else "disabled")
        self.main_notebook.tab(3, state="normal" if session_ready else "disabled")
        self.target_notebook.tab(1, state="normal" if mask_ready else "disabled")

    def _load_config_editors(self) -> None:
        self._load_text_into_widget(self.stream_config_text, Path(self.stream_config_path_var.get()))
        self._load_text_into_widget(self.pp_config_text, Path(self.pp_config_path_var.get()))

    def _load_text_into_widget(self, widget, path: Path) -> None:
        if widget is None:
            return
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            self.set_status(f"Failed to read {path}: {exc}")
            return
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _start_continuous_pointcloud_updates(self) -> None:
        self._continuous_viewer_updates_enabled = True
        self._schedule_continuous_pointcloud_update(initial=True)

    def _stop_continuous_pointcloud_updates(self) -> None:
        self._continuous_viewer_updates_enabled = False
        if self._pointcloud_after_id is not None:
            try:
                self.root.after_cancel(self._pointcloud_after_id)
            except Exception:
                pass
            self._pointcloud_after_id = None

    def _schedule_continuous_pointcloud_update(self, initial: bool = False) -> None:
        if not self._continuous_viewer_updates_enabled:
            return
        delay_ms = 0 if initial else self.CONTINUOUS_POINTCLOUD_UPDATE_MS
        if self._pointcloud_after_id is not None:
            try:
                self.root.after_cancel(self._pointcloud_after_id)
            except Exception:
                pass
        self._pointcloud_after_id = self.root.after(delay_ms, self._continuous_pointcloud_update_tick)

    def _continuous_pointcloud_update_tick(self) -> None:
        self._pointcloud_after_id = None
        if not self._continuous_viewer_updates_enabled or not self.controller.is_viewer_running():
            return
        if self._pointcloud_update_in_progress:
            self._schedule_continuous_pointcloud_update()
            return

        self._pointcloud_update_in_progress = True

        def work():
            self._apply_settings_from_vars()
            return self.controller.compute_point_cloud(
                publish_ros_pointcloud=self.publish_ros_pointcloud_var.get(),
                publish_aruco_markers=self.publish_aruco_markers_var.get(),
                publish_target_position=self.publish_target_position_var.get(),
                viewer_options=self._viewer_options(),
            )

        def on_success(result) -> None:
            self._pointcloud_update_in_progress = False
            self._update_viewer_status()
            self.pointcloud_status_var.set(result.status_message)
            self.set_status(result.status_message)
            self._schedule_continuous_pointcloud_update()

        def on_error(exc) -> None:
            self._pointcloud_update_in_progress = False
            self._schedule_continuous_pointcloud_update()

        self._run_background(
            "Updating point cloud continuously...",
            work,
            on_success=on_success,
            on_error=on_error,
        )

    def _apply_settings_from_vars(self) -> None:
        self.controller.paths = AppPaths(
            stream_config_path=Path(self.stream_config_path_var.get()),
            pp_pointcloud_config_path=Path(self.pp_config_path_var.get()),
            colmap_path=Path(self.colmap_path_var.get()),
            sam_checkpoint=Path(self.sam_checkpoint_var.get()),
            sam_model_cfg=Path(self.sam_model_cfg_var.get()),
        )
        self.controller.options = RuntimeOptions(
            skip_seconds=float(self.skip_seconds_var.get()),
            startup_timeout=float(self.startup_timeout_var.get()),
            point_size=float(self.point_size_var.get()),
            marker_size=float(self.marker_size_var.get()),
        )

    def _viewer_options(self) -> ViewerOptions:
        return ViewerOptions(
            port=int(self.viser_port_var.get()),
            show_aruco_markers=self.show_aruco_in_viewer_var.get(),
            show_target_bbox=self.show_target_bbox_in_viewer_var.get(),
            show_camera_positions=self.show_cameras_in_viewer_var.get(),
            point_size=float(self.point_size_var.get()),
            marker_size=float(self.marker_size_var.get()),
        )

    def _run_background(self, start_message: str, work, on_success=None, on_error=None) -> None:
        self.set_status(start_message)

        def runner() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                details = traceback.format_exc()
                self.root.after(
                    0,
                    lambda error=exc, error_details=details, callback=on_error: self._handle_background_error(
                        error,
                        error_details,
                        callback,
                    ),
                )
                return
            self.root.after(
                0,
                lambda value=result, callback=on_success: self._handle_background_success(value, callback),
            )

        threading.Thread(target=runner, daemon=True).start()

    def _handle_background_success(self, result, on_success) -> None:
        if on_success is not None:
            on_success(result)

    def _handle_background_error(self, exc: Exception, details: str, on_error) -> None:
        self.set_status(f"Error: {exc}")
        if on_error is not None:
            on_error(exc)
        else:
            messagebox.showerror("Fast4DGS", f"{exc}\n\n{details}")

    def on_reload_stream_config(self) -> None:
        self._load_text_into_widget(self.stream_config_text, Path(self.stream_config_path_var.get()))
        self.set_status("Reloaded stream_config.json from disk.")

    def on_reload_pp_config(self) -> None:
        self._load_text_into_widget(self.pp_config_text, Path(self.pp_config_path_var.get()))
        self.set_status("Reloaded pp_pointcloud.json from disk.")

    def on_save_stream_config(self) -> None:
        try:
            self.controller.save_json_text(Path(self.stream_config_path_var.get()), self.stream_config_text.get("1.0", tk.END))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save stream config", str(exc))
            return
        self.set_status("Saved stream_config.json.")

    def on_save_pp_config(self) -> None:
        try:
            self.controller.save_json_text(Path(self.pp_config_path_var.get()), self.pp_config_text.get("1.0", tk.END))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save point-cloud config", str(exc))
            return
        self.set_status("Saved pp_pointcloud.json.")

    def on_save_and_start(self) -> None:
        def work():
            self._apply_settings_from_vars()
            self.controller.save_json_text(self.controller.paths.stream_config_path, self.stream_config_text.get("1.0", tk.END))
            self.controller.save_json_text(self.controller.paths.pp_pointcloud_config_path, self.pp_config_text.get("1.0", tk.END))
            return self.controller.start_session()

        def on_success(batch) -> None:
            display_names = list(batch.display_names)
            self._update_stream_lists(display_names)
            self._update_tab_states(session_ready=True, mask_ready=False)
            self.main_notebook.select(1)
            self.refresh_target_preview()
            self.refresh_individual_stream_view()
            self.refresh_all_streams_view()
            self.set_status(f"Session started. {len(display_names)} streams are ready.")

        self._run_background("Saving configs and starting dataset...", work, on_success=on_success)

    def _update_stream_lists(self, display_names: list[str]) -> None:
        if self.target_stream_combo is not None:
            self.target_stream_combo["values"] = display_names
            if display_names and not self.target_stream_combo.get():
                self.target_stream_combo.set(display_names[0])
        if self.individual_stream_combo is not None:
            self.individual_stream_combo["values"] = display_names
            if display_names and not self.individual_stream_combo.get():
                self.individual_stream_combo.set(display_names[0])

    def _image_to_photo(self, image, max_width: int, max_height: int):
        if image is None:
            return None, 1.0
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        display_image, scale = resize_for_display(image, max_width=max_width, max_height=max_height)
        rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb)), scale

    def refresh_target_preview(self) -> None:
        if self.target_stream_combo is None or not self.target_stream_combo.get():
            return
        try:
            frame = self.controller.get_single_stream_frame(self.target_stream_combo.get())
        except Exception:
            return
        self.current_preview_display_name = self.target_stream_combo.get()
        self.select_original_frame = frame
        self.current_raw_box = None
        self.drag_start = None

        photo, scale = self._image_to_photo(frame, max_width=1050, max_height=700)
        if self.select_canvas is None or photo is None:
            return
        self.select_canvas_scale = scale
        self.select_canvas_photo = photo
        self.select_canvas.delete("all")
        self.select_canvas.config(width=photo.width(), height=photo.height())
        self.select_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.select_rectangle_id = None
        self.selected_overlay_label.configure(image="")
        self._selected_overlay_photo = None

    def clear_box_selection(self) -> None:
        self.current_raw_box = None
        self.drag_start = None
        if self.select_canvas is not None and self.select_rectangle_id is not None:
            self.select_canvas.delete(self.select_rectangle_id)
            self.select_rectangle_id = None
        self.set_status("Cleared target bounding box.")

    def on_canvas_press(self, event) -> None:
        if self.select_canvas is None or self.select_original_frame is None:
            return
        self.drag_start = (event.x, event.y)
        if self.select_rectangle_id is not None:
            self.select_canvas.delete(self.select_rectangle_id)
        self.select_rectangle_id = self.select_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff00", width=2)

    def on_canvas_drag(self, event) -> None:
        if self.select_canvas is None or self.select_rectangle_id is None or self.drag_start is None:
            return
        self.select_canvas.coords(self.select_rectangle_id, self.drag_start[0], self.drag_start[1], event.x, event.y)

    def on_canvas_release(self, event) -> None:
        if self.drag_start is None or self.select_original_frame is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)
        if abs(right - left) < 4 or abs(bottom - top) < 4:
            self.current_raw_box = None
            self.set_status("Bounding box was too small. Draw again.")
            return
        self.current_raw_box = (
            int(round(left / self.select_canvas_scale)),
            int(round(top / self.select_canvas_scale)),
            int(round(right / self.select_canvas_scale)),
            int(round(bottom / self.select_canvas_scale)),
        )
        self.set_status(f"Selected box: {self.current_raw_box}")

    def on_refresh_target_snapshot(self) -> None:
        def work():
            batch = self.controller.refresh_snapshot(require_all_streams=True)
            return list(batch.display_names)

        def on_success(display_names: list[str]) -> None:
            self._update_stream_lists(display_names)
            self.refresh_target_preview()
            self.set_status("Loaded a fresh snapshot for target selection.")

        self._run_background("Refreshing target-selection snapshot...", work, on_success=on_success)

    def on_run_sam(self) -> None:
        if self.current_preview_display_name is None or self.current_raw_box is None:
            messagebox.showwarning("Run SAM", "Select a stream and draw a bounding box first.")
            return

        def work():
            return self.controller.segment_target(self.current_preview_display_name, self.current_raw_box)

        def on_success(state) -> None:
            overlay_photo, _ = self._image_to_photo(state.overlay_grid, max_width=1400, max_height=800)
            selected_photo, _ = self._image_to_photo(state.selected_overlay, max_width=360, max_height=280)
            if overlay_photo is not None:
                self._mask_overlay_photo = overlay_photo
                self.mask_overlay_label.configure(image=overlay_photo)
            if selected_photo is not None:
                self._selected_overlay_photo = selected_photo
                self.selected_overlay_label.configure(image=selected_photo)
            self._update_tab_states(session_ready=True, mask_ready=True)
            self.target_notebook.select(1)
            self.set_status("SAM mask generation finished. Review the masks, then save them to memory.")

        self._run_background("Running SAM on the latest snapshots...", work, on_success=on_success)

    def on_save_mask_to_memory(self) -> None:
        if self.controller.target_state is None:
            messagebox.showwarning("Save mask", "Run SAM first.")
            return
        try:
            self.controller.save_target_to_memory()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save mask", str(exc))
            return
        self.set_status(
            "Saved the mask and its source RGB snapshots to memory. The next point-cloud update will lock the target box and then release the saved mask/images."
        )

    def on_reselect_mask(self) -> None:
        self.controller.clear_target()
        self._update_tab_states(session_ready=True, mask_ready=False)
        self.selected_overlay_label.configure(image="")
        self.mask_overlay_label.configure(image="")
        self._selected_overlay_photo = None
        self._mask_overlay_photo = None
        self.target_notebook.select(0)
        self.clear_box_selection()
        self.set_status("Cleared target mask. Select a new bounding box.")

    def refresh_all_streams_view(self) -> None:
        try:
            batch = self.controller.refresh_snapshot(require_all_streams=False)
        except Exception:
            return
        if not batch.display_names:
            return
        try:
            grid = self.controller.build_all_streams_grid()
        except Exception:
            return
        photo, _ = self._image_to_photo(grid, max_width=1500, max_height=820)
        if photo is not None and self.all_streams_label is not None:
            self._all_streams_photo = photo
            self.all_streams_label.configure(image=photo)

    def refresh_individual_stream_view(self) -> None:
        if self.individual_stream_combo is None or not self.individual_stream_combo.get():
            return
        try:
            frame = self.controller.get_single_stream_frame(self.individual_stream_combo.get())
        except Exception:
            return
        photo, _ = self._image_to_photo(frame, max_width=1200, max_height=800)
        if photo is not None and self.individual_stream_label is not None:
            self._individual_stream_photo = photo
            self.individual_stream_label.configure(image=photo)

    def _schedule_stream_refresh(self) -> None:
        self.root.after(700, self._poll_stream_views)

    def _poll_stream_views(self) -> None:
        if self.controller.dataset is not None:
            self.refresh_all_streams_view()
            self.refresh_individual_stream_view()
        self._schedule_stream_refresh()

    def _update_viewer_status(self) -> None:
        url = self.controller.viewer_url()
        if self.controller.is_viewer_running() and url:
            self.viewer_status_var.set(f"Running at {url}")
        else:
            self.viewer_status_var.set("Viser stopped")

    def on_open_viser(self) -> None:
        def work():
            self._apply_settings_from_vars()
            self.controller.ensure_viewer(int(self.viser_port_var.get()))
            return self.controller.compute_point_cloud(
                publish_ros_pointcloud=self.publish_ros_pointcloud_var.get(),
                publish_aruco_markers=self.publish_aruco_markers_var.get(),
                publish_target_position=self.publish_target_position_var.get(),
                viewer_options=self._viewer_options(),
            )

        def on_success(result) -> None:
            self._update_viewer_status()
            self.pointcloud_status_var.set(result.status_message)
            if result.viewer_url:
                webbrowser.open(result.viewer_url)
            self._start_continuous_pointcloud_updates()
            self.set_status(result.status_message)

        self._run_background("Starting Viser and computing point cloud...", work, on_success=on_success)

    def on_update_point_cloud(self) -> None:
        def work():
            self._apply_settings_from_vars()
            return self.controller.compute_point_cloud(
                publish_ros_pointcloud=self.publish_ros_pointcloud_var.get(),
                publish_aruco_markers=self.publish_aruco_markers_var.get(),
                publish_target_position=self.publish_target_position_var.get(),
                viewer_options=self._viewer_options(),
            )

        def on_success(result) -> None:
            self._pointcloud_update_in_progress = False
            self._update_viewer_status()
            self.pointcloud_status_var.set(result.status_message)
            self.set_status(result.status_message)

        self._pointcloud_update_in_progress = True

        def on_error(exc) -> None:
            self._pointcloud_update_in_progress = False

        self._run_background("Updating point cloud...", work, on_success=on_success, on_error=on_error)

    def on_stop_viser(self) -> None:
        self._stop_continuous_pointcloud_updates()
        self._pointcloud_update_in_progress = False
        self.controller.stop_viewer()
        self._update_viewer_status()
        self.set_status("Stopped Viser.")

    def on_open_viewer_in_browser(self) -> None:
        url = self.controller.viewer_url()
        if not url:
            messagebox.showinfo("Open viewer", "Viser is not running yet.")
            return
        webbrowser.open(url)
        self.set_status(f"Opened {url} in the browser.")

    def on_close(self) -> None:
        try:
            self._stop_continuous_pointcloud_updates()
            self.controller.shutdown()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabbed Fast4DGS GUI")
    parser.add_argument("--stream-config", type=Path, default=None, help="Optional override for stream_config.json")
    parser.add_argument("--pp-config", type=Path, default=None, help="Optional override for pp_pointcloud.json")
    parser.add_argument("--colmap-path", type=Path, default=None, help="Optional override for COLMAP sparse model path")
    parser.add_argument("--sam-checkpoint", type=Path, default=None, help="Optional override for SAM checkpoint path")
    parser.add_argument("--sam-model-cfg", type=Path, default=None, help="Optional override for SAM model config path")
    parser.add_argument("--skip-seconds", type=float, default=None, help="Override startup skip duration")
    parser.add_argument("--startup-timeout", type=float, default=None, help="Override stream startup timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = Fast4DGSController()

    if args.stream_config is not None:
        controller.paths.stream_config_path = args.stream_config
    if args.pp_config is not None:
        controller.paths.pp_pointcloud_config_path = args.pp_config
    if args.colmap_path is not None:
        controller.paths.colmap_path = args.colmap_path
    if args.sam_checkpoint is not None:
        controller.paths.sam_checkpoint = args.sam_checkpoint
    if args.sam_model_cfg is not None:
        controller.paths.sam_model_cfg = args.sam_model_cfg
    if args.skip_seconds is not None:
        controller.options.skip_seconds = args.skip_seconds
    if args.startup_timeout is not None:
        controller.options.startup_timeout = args.startup_timeout

    root = tk.Tk()
    Fast4DGSGui(root, controller)
    root.mainloop()


if __name__ == "__main__":
    main()
