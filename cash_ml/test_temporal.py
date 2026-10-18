import numpy as np
import pytest
import torch

from cash_ml.temporal_model import (
    TemporalHead,
    evaluate_logits,
    load_head,
    point_loss,
    save_head,
    threshold_for,
    training_scaler,
)


def test_unknown_positive_windows_have_zero_direct_loss_gradient():
    binary = torch.zeros(5, requires_grad=True)
    typed = torch.zeros((5, 2), requires_grad=True)
    record = {'label':1, 'annotations':[{'timecode':2.,'type':'a'}]}
    item = {'centers':np.arange(5.)}
    b,t = point_loss(binary,typed,record,item,['a','b'])
    (b+t).backward()
    assert binary.grad.tolist() == [0.,0.,-.5,0.,0.]
    assert torch.count_nonzero(typed.grad[[0,1,3,4]]) == 0


def test_all_normal_windows_receive_negative_supervision():
    binary=torch.zeros(4,requires_grad=True)
    typed=torch.zeros((4,2),requires_grad=True)
    b,_=point_loss(binary,typed,{'label':0,'annotations':[]},{'centers':np.arange(4.)},['a','b'])
    b.backward()
    assert torch.all(binary.grad > 0)


def test_multiple_point_types_are_preserved_at_same_clip():
    b,t=point_loss(torch.zeros(3),torch.zeros((3,2)),
                   {'label':1,'annotations':[{'timecode':1.,'type':'a'},{'timecode':1.,'type':'b'}]},
                   {'centers':np.arange(3.)},['a','b'])
    assert b.item() == pytest.approx(np.log(2))
    assert t.item() == pytest.approx(np.log(2))


@pytest.mark.parametrize('family',['linear','tcn','gru'])
@pytest.mark.parametrize('length',[1,2,17])
def test_head_retains_real_sequence_length(family,length):
    head=TemporalHead(family,dimensions=4,classes=2).eval()
    binary,types=head(torch.zeros((length,4)))
    assert binary.shape == (length,)
    assert types.shape == (length,2)
    assert torch.isfinite(binary).all()


def test_scaler_uses_only_given_training_records():
    data={'train':{'features':np.array([[1.,2.],[3.,4.]])},
          'test':{'features':np.full((20,2),1e10)}}
    mean,_=training_scaler([{'id':'train'}],data)
    assert mean.tolist()==[2.,3.]


def test_saved_head_produces_identical_logits(tmp_path):
    torch.manual_seed(17)
    m=TemporalHead('tcn',4,2).eval()
    x=torch.randn(8,4)
    path=tmp_path/'head.pt'
    save_head(path,m,np.zeros(4,dtype='float32'),np.ones(4,dtype='float32'),.4,
              {'family':'tcn','allowed_types':['a','b']})
    loaded,_,_,threshold,meta=load_head(path)
    assert threshold==.4 and meta['allowed_types']==['a','b']
    for a,b in zip(m(x),loaded(x),strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_threshold_ignores_extra_test_predictions():
    predictions={'p':(np.array([2.]),None),'n':(np.array([-2.]),None),'test':(np.array([1000.]),None)}
    records=[{'id':'p','label':1},{'id':'n','label':0}]
    expected=threshold_for(predictions,records)
    del predictions['test']
    assert threshold_for(predictions,records)==expected


def test_evaluation_preserves_type_and_real_point_coordinates():
    records=[{'id':'a','label':1,'scene_group':'g1','annotations':[{'timecode':2.1,'type':'b'}]},
             {'id':'n','label':0,'scene_group':'g2','annotations':[]}]
    data={r['id']:{'centers':np.array([.1,2.1]),'starts':np.array([0.,1.6]),
                   'ends':np.array([.6,2.5]),'duration':np.asarray(2.6)} for r in records}
    pred={'a':(np.array([-5.,5.]),np.array([[3.,0.],[0.,3.]])),
          'n':(np.array([-5.,-5.]),np.array([[3.,0.],[3.,0.]]))}
    result=evaluate_logits(pred,records,data,.5,['a','b'])
    assert result['detection']['accuracy']==1.
    assert result['timing']['typed_detected_points']==1
    assert result['videos'][0]['events'][0]['timecode_sec']==2.1
    assert result['false_alerts_per_hour'] is None


def test_last_block_freezes_batchnorm_statistics():
    from cash_ml.temporal_finetune import LastBlock
    # Lightweight real BatchNorm exercises the overridden train transition.
    model=LastBlock.__new__(LastBlock)
    torch.nn.Module.__init__(model)
    model.layer4=torch.nn.Sequential(torch.nn.Conv3d(2,2,1),torch.nn.BatchNorm3d(2))
    before=model.layer4[1].running_mean.clone()
    model.train()
    model.layer4(torch.randn(2,2,2,2,2))
    torch.testing.assert_close(before,model.layer4[1].running_mean)
    assert model.layer4[0].weight.requires_grad
    assert not model.layer4[1].weight.requires_grad
def test_prefix_cache_preserves_clip_axis_and_spatial_shape():
    import numpy as np

    from cash_ml.temporal_finetune import prefix_tensor

    original = np.arange(2 * 256 * 4 * 14 * 14, dtype=np.float32).reshape(2, -1)
    corrected = prefix_tensor(original.reshape(-1, 256, 4, 7, 7), 2)
    assert corrected.shape == (2, 256, 4, 14, 14)
    np.testing.assert_array_equal(corrected.reshape(2, -1), original)
