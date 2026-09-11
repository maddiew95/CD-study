import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier


class LSTM(nn.Module):
    """LSTM backbone used as (a) classifier during stage 1, and
    (b) feature extractor for XGBoost during stage 2."""
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32):
        super().__init__()
        # Input (B,1,13) -> (B, 13, 1): 13 timesteps, 1 feature each
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)   # used in stage 1 only

    def features(self, x):                # x: (B, 1, 13)
        x = x.transpose(1, 2)             # (B, 13, 1)
        _, (h_n, _) = self.lstm(x)        # (1, B, 32)
        return h_n.squeeze(0)             # (B, 32)

    def forward(self, x):
        return self.head(self.features(x))


class LSTM_XGB:
    """
    Two-stage hybrid:
      Stage 1: train LSTM end-to-end with softmax head (supervised)
      Stage 2: freeze LSTM, extract 32-D features, fit XGBoost on them
    """
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32,
                 device: str = "cpu", seed: int = 0):
        self.device = device
        torch.manual_seed(seed)
        self.backbone = LSTM(n_classes, lstm_hidden).to(device)
        # Xavier init on the linear head (LSTM uses its own init)
        nn.init.xavier_uniform_(self.backbone.head.weight)
        nn.init.zeros_(self.backbone.head.bias)

        self.xgb = XGBClassifier(
            # --- Verified from Li Table IV ---
            booster="gbtree",              # Li: booster = gbtree
            learning_rate=0.1,             # Li: eta = 0.1
            n_estimators=56,               # Li: iters_optimal = 56
            # --- Corrected from Li's typo ---
            objective="multi:softprob",    # Li listed reg:linear (typo -- incompatible with 8-class)
            num_class=n_classes,
            # --- Standard defaults (not specified by Li) ---
            max_depth=6,                   # XGBoost default; Li doesn't publish
            subsample=1.0,                 # default
            colsample_bytree=1.0,          # default
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
        )


    def fit_lstm(self, X_train, y_train, X_val=None, y_val=None,
                 epochs=70, lr=1e-2, weight_decay=1e-4,
                 batch_size=50, seed=0, verbose=False):
        opt = optim.Adam(self.backbone.parameters(), lr=lr,
                         betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        # MATLAB-style shuffle once
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X_train), generator=g)
        X_s, y_s = X_train[perm], y_train[perm]
        loader = DataLoader(TensorDataset(X_s, y_s),
                            batch_size=batch_size, shuffle=False)

        for ep in range(1, epochs + 1):
            self.backbone.train()
            tr_loss = tr_correct = tr_n = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.backbone(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tr_loss    += loss.item() * xb.size(0)
                tr_correct += (logits.argmax(1) == yb).sum().item()
                tr_n       += xb.size(0)
            scheduler.step()

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                msg = f"  [stage1] ep {ep:3d} | loss {tr_loss/tr_n:.4f} acc {tr_correct/tr_n:.3f}"
                if X_val is not None and y_val is not None:
                    va_loss, va_acc = self._eval_backbone(X_val, y_val, loss_fn)
                    msg += f" | val loss {va_loss:.4f} acc {va_acc:.3f}"
                print(msg)

    @torch.no_grad()
    def _eval_backbone(self, X, y, loss_fn):
        self.backbone.eval()
        logits = self.backbone(X.to(self.device))
        loss = loss_fn(logits, y.to(self.device)).item()
        acc = (logits.argmax(1) == y.to(self.device)).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def _extract(self, X):
        self.backbone.eval()
        return self.backbone.features(X.to(self.device)).cpu().numpy()

    def fit_xgb(self, X, y):
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        self.xgb.fit(self._extract(X), y_np)

    def predict(self, X):
        return self.xgb.predict(self._extract(X))