(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        yellow - block
        red - block
        purple - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (clear blue)
        (ontable green)
        (clear green)
        (ontable yellow)
        (on red yellow)
        (on purple red)
        (clear purple)
        (ontable orange)
        (clear orange)
        ; blue
        ; green
        ; yellow -> red -> purple
        ; orange
    )
    (:goal (and (on yellow green)
        (on blue yellow)
        (on red blue)
        (on purple red)
        (on orange purple)))
)
    