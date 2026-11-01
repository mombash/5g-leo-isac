function occ = occupancy_mask(cfg, mode, seed)
%OCCUPANCY_MASK PDSCH PRB/slot mask for reference-only, low, and full load.

    arguments
        cfg struct
        mode (1,1) string {mustBeMember(mode,["full","low_15","reference_only","sparse"])}
        seed (1,1) double = 0
    end
    nRB=cfg.n_rb; nSlot=cfg.n_slot_frame;
    occ=struct(); occ.mode=mode; occ.seed=seed; occ.period=nSlot;
    occ.slotAllocation=0:(nSlot-1); occ.mask=false(nRB,nSlot); occ.pdsch_enable=true;
    switch mode
        case "full"
            occ.prbSet=0:(nRB-1);
        case "low_15"
            occ.prbSet=unique(round(linspace(0,nRB-1,8)));
        case "reference_only"
            occ.prbSet=[]; occ.slotAllocation=[]; occ.pdsch_enable=false;
        case "sparse" % 25% PRB occupancy
            occ.prbSet=0:4:(nRB-1);
    end
    if occ.pdsch_enable, occ.mask(occ.prbSet+1,1:nSlot)=true; end
    occ.rho=nnz(occ.mask)/numel(occ.mask);
    occ.power_policy="constant_active_RE_amplitude";
end
