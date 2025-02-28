import numpy as np

def histcounts2(x, bins=None, edges=None,
               BinLimits=None, BinWidth=None, BinMethod=None,
               Normalization=None,  # 'count', 'countdensity', 'cumcount', 'probability', 'pdf', 'cdf', 'percentage'
               return_bin_indices=False):
    """
    Repliziert (ungefähr) das Verhalten von Matlab histcounts für numerische Daten.

    Syntax-Anlehnung:
        n, edges = histcounts(x)                          # automatische Bin-Auswahl
        n, edges = histcounts(x, bins)                    # wenn bins int -> Anzahl Bins
                                                          # wenn bins array -> Bin-Kanten
        n, edges, bin = histcounts(..., return_bin_indices=True)
        # via "Name=Value" kann man BinLimits, BinWidth, BinMethod, Normalization etc. übergeben.

    Wichtige Unterschiede zum reinen np.histogram:
      - Matlab default: Bins sind links-inklusive, rechts-exklusiv, das letzte Bin ist rechts-inklusive
      - Hier versuchen wir, dieses Verhalten nachzuahmen.
    """

    # --- Vorbereitung & Typ-Prüfungen ---
    if not (isinstance(x, (np.ndarray, list, tuple))):
        raise TypeError("x must be array-like numeric data.")

    x = np.asarray(x, dtype=float)  # konvertiere nach float
    # Filter out NaN/Inf aus der Min/Max-Betrachtung, s. Matlab-Code
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]  # nur für min/max Tests

    if x_finite.size == 0:
        # Falls alles nur NaN/Inf ist, definieren wir edges rudimentär:
        minx, maxx = 0., 1.
    else:
        minx, maxx = np.min(x_finite), np.max(x_finite)

    # Falls gar nichts angegeben ist, bins=None, edges=None etc. -> Default 'auto'
    # Falls bins gegeben ist, kann es eine int (nBins) oder ein Array (BinEdges) sein
    #   -> Nach Matlab-Logik: wenn bins kein Skalar und >1 Element => interpretieren als edges
    #                        wenn bins ein Skalar => anzahl bins
    # (Der Matlab-Code unterscheidet das ebenfalls.)
    if bins is not None and edges is None:
        if np.isscalar(bins):
            # bins = Anzahl Bins
            # Matlab: edges = binpicker(minx, maxx, bins, ???)
            edges = _manual_binpicker(minx, maxx, int(bins))
        else:
            # bins = array => interpretieren als bin edges
            edges = np.asarray(bins, dtype=float)

    # Falls edges explizit übergeben wurde (inkl. obiges), nutzen wir das.
    # Ansonsten schauen wir nach BinWidth, BinMethod, etc.
    if edges is None:
        # A) BinLimits?
        if BinLimits is not None:
            # in Matlab: [minx, maxx] = BinLimits
            # Dann z.B. 'NumBins', 'BinWidth' oder 'BinMethod' => generiert edges
            if len(BinLimits) != 2:
                raise ValueError("BinLimits must be a 2-element array [low, high].")
            lim_min, lim_max = BinLimits
            # Min/Max anpassen (Matlab schneidet Daten außerhalb dieser Limits ab)
            # aber hier nur fürs Erzeugen der Kanten
            if BinWidth is not None:
                edges = _edges_via_binwidth(lim_min, lim_max, BinWidth)
            elif bins is not None:  # => interpretieren als NumBins
                edges = _edges_via_nbins(lim_min, lim_max, bins)
            else:
                # => BinMethod
                chosen_method = BinMethod.lower() if BinMethod else 'auto'
                edges = _edges_via_binmethod(x_finite[(x_finite >= lim_min) & (x_finite <= lim_max)],
                                             lim_min, lim_max,
                                             chosen_method)
        else:
            # B) kein BinLimits
            # => Entweder BinEdges von 'bins' oder 'NumBins' (wenn bins int),
            #    oder BinWidth, oder BinMethod
            if BinWidth is not None:
                edges = _edges_via_binwidth(minx, maxx, BinWidth)
            elif bins is not None:  # => interpretieren als NumBins
                edges = _edges_via_nbins(minx, maxx, bins)
            else:
                # => BinMethod
                chosen_method = BinMethod.lower() if BinMethod else 'auto'
                edges = _edges_via_binmethod(x_finite, minx, maxx, chosen_method)

    # Jetzt haben wir edges.
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1:
        raise ValueError("Bin edges must be a 1D array.")
    if len(edges) < 2:
        raise ValueError("Need at least 2 bin edges.")

    # Matlab 'histcounts' ist i.d.R. linksinklusive, rechtes Ende exklusive
    # das letzte Bin aber beidseitig inklusiv. In NumPy 1.11+ kann man
    # np.histogram(x, bins=edges, right=False) nutzen. Dann ist
    #   bin i: [edges[i], edges[i+1])  (letzter ist [edges[-2], edges[-1])) – exklusive
    # Wir wollen aber, dass der letzte Bin edges[-1] inklusive ist, also manuell:

    # => zähle, in welchen Intervall x liegt (manuell):
    # digitize: gibt bin-Index (1-based), so dass edges[i-1] <= x < edges[i], EXCEPT
    # digitize nutzt standardmäßig right=False => also [edge[i-1], edge[i]).
    # Dann machen wir: bin_idx = np.digitize(x, edges, right=False)
    bin_idx = np.digitize(x, edges, right=False)

    # In Matlab:
    #  - Werte == edges[-1] sollen ins letzte Bin fallen (bin_idx = len(edges)-1)
    # digitize() mit right=False tut das NICHT (es packt x==edges[-1] nach bin=len(edges)).
    # => wir korrigieren manuell:
    last_edge = edges[-1]
    # Alle, die == last_edge sind, sollen bin_idx = len(edges)-1 bekommen:
    eq_last_edge_mask = (x == last_edge)
    bin_idx[eq_last_edge_mask] = len(edges) - 1

    # Alles, was < edges[0], kriegt bin_idx=0 => das ist "links außerhalb"
    # Alles, was >= edges[-1], kriegt bin_idx=len(edges) => "rechts außerhalb"
    # (Matlab wirft die außerhalb-liegenden x aus dem Histogramm.)
    # => also wir wollen nur 1..(len(edges)-1) als "gültige" Bins
    valid_mask = (bin_idx >= 1) & (bin_idx <= (len(edges)-1))
    # build histogram
    # bin_idx - 1 => zero-based Bins. So in Python: Bin #0 => [edges[0], edges[1])
    #                                  Bin #1 => [edges[1], edges[2]) ...
    #                                  Bin #(k-1) => [edges[k-1], edges[k])
    # => in Matlab hat man n Bins = len(edges)-1
    counts = np.zeros(len(edges) - 1, dtype=float)
    # wir inkrementieren die Zähler (bin_idx-1) nur für gültige:
    np.add.at(counts, bin_idx[valid_mask] - 1, 1)

    # Normalization
    if Normalization is not None:
        Normalization = Normalization.lower()
        if Normalization == 'count':
            pass  # nichts machen
        elif Normalization == 'countdensity':
            # n ./ diff(edges)
            bin_widths = np.diff(edges)
            counts = counts / bin_widths
        elif Normalization == 'cumcount':
            counts = np.cumsum(counts)
        elif Normalization == 'probability':
            counts = counts / x.size
        elif Normalization == 'pdf':
            bin_widths = np.diff(edges)
            counts = counts / x.size / bin_widths
        elif Normalization == 'cdf':
            counts = np.cumsum(counts / x.size)
        elif Normalization == 'percentage':
            counts = (100.0 * counts) / x.size
        else:
            raise ValueError(f"Unknown normalization: {Normalization}")

    # Falls man die bin-Indices wie in Matlab zurück will
    # => in Matlab ist BIN(k) der Bin-Index für X(k), oder 0 falls X(k) nicht in Edges liegt
    # Wir müssen also invalid in 0 konvertieren:
    if return_bin_indices:
        final_bin_idx = np.where(valid_mask, bin_idx, 0)  # 1..(len(edges)-1), oder 0
        return counts, edges, final_bin_idx
    else:
        return counts, edges


# ----------------------------------------------------------------------------------
# Hilfsfunktionen zur (ungefähren) Replikation der Matlab-Bin-Methoden
# ----------------------------------------------------------------------------------

def _manual_binpicker(minx, maxx, nbins):
    """
    Entspricht in Matlab ungefähr:
      edges = binpicker(minx,maxx,nbins,(maxx-minx)/nbins)
    """
    nbins = int(nbins)
    if nbins < 1:
        nbins = 1
    if minx == maxx:
        # in Matlab haben wir dann einfach edges=[minx; minx+1]
        return np.array([minx, minx + 1.0])
    step = (maxx - minx) / nbins if nbins > 0 else 1.0
    # edges = [minx, minx+step, ..., maxx]
    return np.linspace(minx, maxx, nbins+1)


def _edges_via_nbins(minx, maxx, nbins):
    return _manual_binpicker(minx, maxx, nbins)


def _edges_via_binwidth(minx, maxx, binwidth):
    binwidth = float(binwidth)
    if binwidth <= 0:
        raise ValueError("BinWidth must be > 0.")
    # In Matlab wird zusätzlich geprüft, ob man max. Bins überschreitet.
    # Wir machen es hier einfach.
    edges = [minx]
    current = minx
    while current < maxx:
        current += binwidth
        # kleine Rundungsfehler abfangen
        if current > maxx and (current - maxx) < 1e-12:
            current = maxx
        edges.append(current)
    if edges[-1] < maxx:
        edges.append(maxx)
    return np.array(edges)


def _edges_via_binmethod(x, minx, maxx, method):
    # Falls x leer oder fast leer -> triviale edges
    if x.size == 0 or minx == maxx:
        return np.array([minx, maxx+1e-9])  # minimal breiter als 0

    method = method.lower()
    if method == 'auto':
        # Matlab "auto": heuristik. Standardmäßig intern "sturges" oder "fd"
        # Je nach Datengröße ...
        # Hier machen wir es einfach:
        return _fdrule(x, minx, maxx, True)
    elif method == 'scott':
        return _scottsrule(x, minx, maxx, True)
    elif method == 'fd':
        return _fdrule(x, minx, maxx, True)
    elif method == 'integers':
        return _integerrule(x, minx, maxx, True, 65536)  # 65536 = getmaxnumbins() in Matlab
    elif method == 'sqrt':
        return _sqrtrule(x, minx, maxx, True)
    elif method == 'sturges':
        return _sturgesrule(x, minx, maxx, True)
    else:
        raise ValueError(f"Unsupported BinMethod: {method}")


def _scottsrule(x, minx, maxx, limit_range):
    """
    Data-based bandwidth rule from Matlab for "scott".
    Bin width ~ 3.5 * std(x) / n^(1/3)
    """
    # exclude inf/NaN already excluded
    N = x.size
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = maxx - minx if (maxx > minx) else 1.0
    bw = 3.5*sigma / (N**(1/3))
    if bw <= 0:
        bw = (maxx - minx) / 10.0  # fallback
    edges = _edges_via_binwidth(minx, maxx, bw)
    return edges


def _fdrule(x, minx, maxx, limit_range):
    """
    Freedman-Diaconis rule
      Bin width ~ 2 * IQR / n^(1/3)
    """
    N = x.size
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-12:
        # fallback
        iqr = np.std(x)
        if iqr < 1e-12:
            iqr = maxx - minx if (maxx > minx) else 1.0
    bw = 2.0 * iqr / (N**(1/3))
    if bw <= 0:
        bw = (maxx - minx) / 10.0
    edges = _edges_via_binwidth(minx, maxx, bw)
    return edges


def _sqrtrule(x, minx, maxx, limit_range):
    """
    Bin width ~ (max-min) / sqrt(N)
    """
    N = x.size
    nbins = int(np.ceil(np.sqrt(N)))
    return _manual_binpicker(minx, maxx, nbins)


def _sturgesrule(x, minx, maxx, limit_range):
    """
    Bin width ~ (max-min) / (log2(N) + 1)
    """
    N = x.size
    nbins = int(np.ceil(np.log2(N) + 1))
    return _manual_binpicker(minx, maxx, nbins)


def _integerrule(x, minx, maxx, limit_range, maxnumbins):
    """
    Matlab's 'integers' method tries to place a bin boundary at every integer.
    Clamps to maxnumbins if too large.
    """
    lower = np.floor(minx)
    upper = np.ceil(maxx)
    nbins = upper - lower
    if nbins < 1:
        nbins = 1
    if nbins > maxnumbins:
        nbins = maxnumbins
    return np.arange(lower, lower + nbins + 1)
