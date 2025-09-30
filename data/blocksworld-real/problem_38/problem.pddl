(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        yellow - block
        red - block
        orange - block
        green - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on yellow purple)
        (on red yellow)
        (clear red)
        (ontable orange)
        (on green orange)
        (clear green)
        (ontable blue)
        (clear blue)
        ; purple -> yellow -> red
        ; orange -> green
        ; blue
    )
    (:goal (and (on blue green)
        (on red blue)
        (on purple red)
        (on yellow purple)
        (on orange yellow)))
)
    