function prepare_manuscript_waveforms(root,split,indices)
    if nargin<3, indices=1:4; end
    project=fileparts(mfilename('fullpath'));
    addpath(fullfile(project,'simulator','matlab'));
    manifest=jsondecode(fileread(fullfile(root,[split '_manifest.json'])));
    for index=indices
        wave=manifest.waveforms(index);path=wave.cache_path;
        metadata=fullfile(root,sprintf('wave_%s_%d.json',split,wave.waveform_id));
        if isfile(path) && isfile(metadata),continue;end
        folder=fileparts(path);if ~isfolder(folder),mkdir(folder);end
        cfg=n256_config();cfg.n_cell_id=wave.cell_id;
        dl=n256_dl_carrier(cfg,occupancy_mask(cfg,"full"),402);
        dl.PDSCH{1}.DataSource={'PN23',double(wave.payload_seed)};
        dl.PDCCH{1}.DataSource={'PN15',double(wave.control_seed)};
        started=tic;[tx,info]=nrWaveformGenerator(dl); %#ok<ASGLU>
        tx=tx(:);native_power=mean(abs(tx).^2);tx=tx/sqrt(native_power);
        assert(numel(tx)==round(4.02*cfg.fs_Hz));
        fid=fopen([path '.tmp'],'w');fwrite(fid,[real(tx).';imag(tx).'],'single');fclose(fid);
        movefile([path '.tmp'],path,'f');
        result=struct('split',split,'waveform_id',wave.waveform_id,'cell_id',wave.cell_id,...
            'payload_seed',wave.payload_seed,'control_seed',wave.control_seed,...
            'samples',numel(tx),'sample_rate_Hz',cfg.fs_Hz,'average_power',mean(abs(tx).^2),...
            'native_power',native_power,'generation_s',toc(started));
        fid=fopen(metadata,'w');fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));fclose(fid);
        fprintf('WAVEFORM %s %d COMPLETE %.1f seconds\n',split,wave.waveform_id,toc(started));
        clear tx info dl;
    end
end
