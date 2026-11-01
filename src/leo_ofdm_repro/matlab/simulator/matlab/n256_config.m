function cfg = n256_config()
%N256_CONFIG Frozen Release-19 n256 downlink constants for the first campaign.
%
%   cfg = n256_config()
%
%   Pure numbers and first-campaign flags. No I/Q. Every other simulator
%   block should take this struct so numerology cannot drift.
%
%   Independent check: nrCarrierConfig with SubcarrierSpacing=30,
%   NSizeGrid=51 must yield SampleRate=30.72e6 and Nfft=1024.

    cfg = struct();

    cfg.band = "n256";
    cfg.duplex = "FDD";
    cfg.dl_band_Hz = [2170e6, 2200e6];
    cfg.fc_Hz = 2.190e9;
    cfg.channel_bw_Hz = 20e6;
    cfg.occupied_bw_Hz = 18.36e6;

    cfg.frequency_range = "FR1";
    cfg.scs_data_kHz = 30;
    cfg.n_rb = 51;
    cfg.n_sc_occupied = 612;
    cfg.n_start_grid = 0;

    cfg.scs_ssb_kHz = 15;
    cfg.ssb_block_pattern = "Case A";
    cfg.ssb_n_rb_15kHz = 20;
    cfg.n_rb_15kHz_in_20MHz = 106;
    cfg.n_rb_ssb_carrier = 102;
    cfg.ssb_ncrb = 41;
    cfg.ssb_transmitted_blocks = [1 0 0 0];
    cfg.ssb_period_ms = 20;

    cfg.fs_Hz = 30.72e6;
    cfg.nfft = 1024;
    cfg.cp_first = 88;
    cfg.cp_other = 72;
    cfg.samples_per_slot = 15360;
    cfg.samples_per_frame = 307200;

    cfg.t_frame_s = 10e-3;
    cfg.t_subframe_s = 1e-3;
    cfg.n_slot_frame = 20;
    cfg.n_slot_subframe = 2;
    cfg.n_symb_slot = 14;
    cfg.cyclic_prefix = "normal";

    cfg.one_bwp = true;
    cfg.n_size_bwp = 51;
    cfg.n_start_bwp = 0;

    cfg.n_cell_id = 1;
    cfg.rnti = 1;
    cfg.pdsch_modulation = "QPSK";
    cfg.pdsch_mapping_type = "A";
    cfg.pdsch_symbol_allocation = [0 14];

    cfg.dmrs_configuration_type = 1;
    cfg.dmrs_typeA_position = 2;
    cfg.dmrs_additional_position = 0;
    cfg.dmrs_length = 1;
    cfg.dmrs_nscid = 0;

    cfg.csirs_enable = false;
    cfg.ptrs_enable = false;
    cfg.windowing_percent = 0;

    cfg.c_mps = 299792458;
    cfg.leakage = 0;
    cfg.clutter_enable = false;
end
