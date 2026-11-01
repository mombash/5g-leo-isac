function result=generate_waveform_smoke(waveform_json,output_path,frames)
    if nargin<3,frames=3;end
    assert(isscalar(frames)&&frames>=1&&frames<=5&&frames==fix(frames),'Smoke is limited to 1–5 NR frames');
    assert(~isfile(output_path),'Existing waveform output is preserved');
    project=fileparts(mfilename('fullpath'));addpath(fullfile(project,'simulator','matlab'));
    wave=jsondecode(fileread(waveform_json));cfg=n256_config();cfg.n_cell_id=wave.cell_id;
    dl=n256_dl_carrier(cfg,occupancy_mask(cfg,"full"),frames);
    dl.PDSCH{1}.DataSource={'PN23',double(wave.payload_seed)};
    dl.PDCCH{1}.DataSource={'PN15',double(wave.control_seed)};
    started=tic;[tx,~]=nrWaveformGenerator(dl);tx=tx(:);
    native_power=mean(abs(tx).^2);tx=tx/sqrt(native_power);
    assert(numel(tx)==round(frames*.01*cfg.fs_Hz));
    fid=fopen(output_path,'w');assert(fid>=0);fwrite(fid,[real(tx).';imag(tx).'],'single');fclose(fid);
    result=struct('scope','Sliced NR smoke; not a recovered 402-frame waveform',...
      'frames',frames,'samples',numel(tx),'sample_rate_Hz',cfg.fs_Hz,...
      'cell_id',wave.cell_id,'payload_seed',wave.payload_seed,'control_seed',wave.control_seed,...
      'native_power',native_power,'average_power',mean(abs(tx).^2),'generation_s',toc(started),...
      'normalization_scope','This short generated waveform only; full-waveform normalization not verified',...
      'matlab_release',version('-release'),'fiveg_license_test_query',logical(license('test','5G_Toolbox')));
    fid=fopen([output_path '.json'],'w');fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));fclose(fid);
end
