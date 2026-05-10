from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts


def fc_settings():
    args = {"section": ("fc", "Forge Couple"), "category_id": "sd"}

    opts.add_option(
        "fc_do_interrupt",
        OptionInfo(
            True,
            "Interrupt on Error",
            **args,
        )
        .info('if disabled, Forge Couple will simply "fail silently"')
        .needs_restart(),
    )

    opts.add_option(
        "fc_no_presets",
        OptionInfo(
            False,
            "Disable the Presets feature in Advanced mode",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_no_tile",
        OptionInfo(
            False,
            "Disable the Tile mode in img2img",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_adv_newline",
        OptionInfo(
            False,
            "Keep newline characters in Advanced mode dataframe",
            **args,
        ).info('newlines would be shown as "\\n" literals'),
    )

    opts.add_option(
        "fc_debug_logging",
        OptionInfo(
            False,
            "Debug logging",
            **args,
        ).info("write structured Forge Couple debug logs to logs/forge_couple_debug.log"),
    )

    opts.add_option(
        "fc_debug_dump_masks",
        OptionInfo(
            False,
            "Dump masks to disk",
            **args,
        ).info("when debug logging is enabled, save cropped masks to logs/forge_couple_masks"),
    )


on_ui_settings(fc_settings)
