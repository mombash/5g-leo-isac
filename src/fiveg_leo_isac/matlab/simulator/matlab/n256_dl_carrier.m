function dl = n256_dl_carrier(cfg, occ, nFrames)
%N256_DL_CARRIER Build the 5G Toolbox n256 downlink object.

    arguments
        cfg struct
        occ struct
        nFrames (1,1) double {mustBeInteger,mustBePositive}=1
    end
    dl=nrDLCarrierConfig(char(cfg.frequency_range),cfg.channel_bw_Hz/1e6,cfg.scs_data_kHz);
    dl.NCellID=cfg.n_cell_id; dl.NumSubframes=10*nFrames;
    dl.CarrierFrequency=cfg.fc_Hz; dl.SampleRate=cfg.fs_Hz; dl.WindowingPercent=cfg.windowing_percent;
    scs30=dl.SCSCarriers{1}; scs30.SubcarrierSpacing=cfg.scs_data_kHz;
    scs30.NSizeGrid=cfg.n_rb; scs30.NStartGrid=cfg.n_start_grid;
    scs15=nrSCSCarrierConfig; scs15.SubcarrierSpacing=cfg.scs_ssb_kHz;
    scs15.NSizeGrid=cfg.n_rb_ssb_carrier; scs15.NStartGrid=0; dl.SCSCarriers={scs30,scs15};
    bwp=dl.BandwidthParts{1}; bwp.SubcarrierSpacing=cfg.scs_data_kHz;
    bwp.CyclicPrefix=char(cfg.cyclic_prefix); bwp.NSizeBWP=cfg.n_size_bwp;
    bwp.NStartBWP=cfg.n_start_bwp; dl.BandwidthParts={bwp};
    ss=dl.SSBurst; ss.Enable=1; ss.BlockPattern=char(cfg.ssb_block_pattern);
    ss.TransmittedBlocks=cfg.ssb_transmitted_blocks; ss.Period=cfg.ssb_period_ms;
    ss.NCRBSSB=cfg.ssb_ncrb; ss.SubcarrierSpacingCommon=cfg.scs_data_kHz; dl.SSBurst=ss;
    pdsch=dl.PDSCH{1}; pdsch.Enable=occ.pdsch_enable;
    pdsch.Modulation=char(cfg.pdsch_modulation); pdsch.MappingType=char(cfg.pdsch_mapping_type);
    pdsch.SymbolAllocation=cfg.pdsch_symbol_allocation;
    if occ.pdsch_enable
        pdsch.SlotAllocation=occ.slotAllocation; pdsch.Period=occ.period; pdsch.PRBSet=occ.prbSet;
    end
    pdsch.RNTI=cfg.rnti; pdsch.EnablePTRS=cfg.ptrs_enable;
    pdsch.DMRS.DMRSConfigurationType=cfg.dmrs_configuration_type;
    pdsch.DMRS.DMRSTypeAPosition=cfg.dmrs_typeA_position;
    pdsch.DMRS.DMRSAdditionalPosition=cfg.dmrs_additional_position;
    pdsch.DMRS.DMRSLength=cfg.dmrs_length; pdsch.DMRS.NSCID=cfg.dmrs_nscid;
    dl.PDSCH={pdsch}; csirs=dl.CSIRS{1}; csirs.Enable=cfg.csirs_enable; dl.CSIRS={csirs};
end

